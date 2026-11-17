from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from driftaware_sialt import workflow
from driftaware_sialt.config_loader import load_config


def _plan(files, **kwargs):
    return workflow.stacking_configs(
        files['stacking'], hemisphere='sh', sensors=['cryosat2'], **kwargs)


def _complete(files, monkeypatch, years='2018:2019'):
    configs = _plan(files, years=years)
    manifest_path, manifest = workflow.prepare_batch(configs, 'local')
    monkeypatch.setattr(workflow, 'init_logger', lambda config: None)
    for entry in manifest['runs']:
        workflow.execute_snapshot(entry['snapshot'], lambda c: (
            Path(c['run']['directory']) / 'final.csv').write_text('fixture'))
    return manifest_path, manifest


def test_selection_is_applied_before_paths_and_projection(workflow_configs):
    config = _plan(workflow_configs, years='2018')[0]
    assert config['options']['hemisphere'] == 'sh'
    assert config['options']['out_epsg'] == 'EPSG:6932'
    assert config['options']['sensor'] == ['cryosat2']
    assert '/sh/cryosat2' in config['input_dir']['cryosat2']
    assert config['output_dir']['trajectories'].endswith('/cryosat2')
    assert 't_start' not in config['stacking'] and 't_length' not in config['stacking']


def test_snapshots_are_self_contained_and_padding_is_not_reapplied(workflow_configs, monkeypatch):
    config = _plan(workflow_configs, years='2018')[0]
    manifest_path, manifest = workflow.prepare_batch([config], 'local')
    snapshot = Path(manifest['runs'][0]['snapshot'])
    assert snapshot.parent == Path(config['run']['directory'])
    workflow_configs['common'].unlink()
    workflow_configs['stacking'].unlink()
    seen = []
    monkeypatch.setattr(workflow, 'init_logger', lambda c: None)

    def stage(frozen):
        seen.append(frozen)
        (snapshot.parent / 'final.csv').write_text('fixture')

    workflow.execute_snapshot(snapshot, stage)

    assert seen[0] == config
    assert seen[0]['run']['processing_start'] == '2017-12-28'
    assert seen[0]['run']['processing_end'] == '2019-01-04'
    assert workflow.read_yaml(snapshot.parent / 'status.yaml')['state'] == 'completed'
    assert workflow.read_yaml(manifest_path)['runs'][0]['snapshot'] == str(snapshot)
    with pytest.raises(FileExistsError):
        workflow.execute_snapshot(snapshot, stage)


def test_snapshots_keep_both_hemispheres_and_periods_independent(workflow_configs):
    sh = _plan(workflow_configs, years='2018:2024')
    nh = workflow.stacking_configs(workflow_configs['stacking'], hemisphere='nh',
                                   sensors=['sentinel3a'], years='2018:2023')
    paths = [c['run']['directory'] for c in sh + nh]
    assert len(set(paths)) == 13
    assert all(not Path(path).exists() for path in paths)
    workflow.prepare_batch(sh, 'slurm')
    workflow.prepare_batch(nh, 'slurm')
    assert all(Path(path, workflow.SNAPSHOT_NAME).is_file() for path in paths)


def test_gridding_infers_inputs_and_freezes_its_own_settings(workflow_configs, monkeypatch):
    path, manifest = _complete(workflow_configs, monkeypatch)
    configs = workflow.gridding_configs(workflow_configs['gridding'], source_batch=path)
    assert len(configs) == 2
    for config, source in zip(configs, manifest['runs']):
        assert config['run']['source_run'] == source['directory']
        assert config['options']['hemisphere'] == 'sh'
        assert config['options']['sensor'] == ['cryosat2']
        assert config['options']['out_epsg'] == 'EPSG:6932'
        assert config['run']['source_run_id'] == source['run_id']
        assert config['run']['processing_start'] == config['run']['output_start']
        assert config['run']['processing_end'] == config['run']['output_end']
        assert 'csv_dir' not in config['gridding']
    grid_path, grid_batch = workflow.prepare_batch(configs, 'local')
    workflow.execute_snapshot(grid_batch['runs'][0]['snapshot'], lambda c: (
        Path(c['run']['directory']) / 'result.nc').write_bytes(b'fixture'))
    assert workflow.read_yaml(Path(configs[0]['run']['directory'], 'status.yaml'))['state'] == 'completed'
    assert workflow.read_yaml(grid_path)['stage'] == 'gridding'


def test_failed_stacking_cannot_be_used_for_gridding(workflow_configs, monkeypatch):
    configs = _plan(workflow_configs, years='2018')
    _, batch = workflow.prepare_batch(configs, 'local')
    monkeypatch.setattr(workflow, 'init_logger', lambda c: None)

    def fail(config):
        (Path(config['run']['directory']) / 'partial.csv').write_text('fixture')
        raise RuntimeError('worker failed')

    with pytest.raises(RuntimeError, match='worker failed'):
        workflow.execute_snapshot(batch['runs'][0]['snapshot'], fail)
    with pytest.raises(ValueError, match='not completed'):
        workflow.gridding_configs(workflow_configs['gridding'], source_run=configs[0]['run']['directory'])


def test_empty_processing_result_is_not_marked_successful(workflow_configs, monkeypatch):
    configs = _plan(workflow_configs, years='2018')
    _, batch = workflow.prepare_batch(configs, 'local')
    monkeypatch.setattr(workflow, 'init_logger', lambda c: None)
    with pytest.raises(RuntimeError, match='produced no'):
        workflow.execute_snapshot(batch['runs'][0]['snapshot'], lambda c: None)
    assert workflow.read_yaml(Path(configs[0]['run']['directory'], 'status.yaml'))['state'] == 'failed'


def test_slurm_receives_frozen_configs_and_records_ids(workflow_configs, monkeypatch, tmp_path):
    configs = _plan(workflow_configs, years='2018:2024')
    path, batch = workflow.prepare_batch(configs, 'slurm')
    script = tmp_path / 'batch script.sh'
    script.write_text('srun driftaware stacking "$1"\n')
    calls = []

    def submit(command, **kwargs):
        assert all(Path(entry['snapshot']).exists() for entry in batch['runs'])
        assert command[-2] == str(script)
        assert workflow.read_snapshot(command[-1])['options']['hemisphere'] == 'sh'
        assert kwargs['check'] and kwargs['capture_output']
        calls.append(command)
        return SimpleNamespace(stdout=f'{100 + len(calls)};cluster\n')

    monkeypatch.setattr(workflow.subprocess, 'run', submit)
    workflow.submit_batch(path, batch, script)
    assert len(calls) == 7
    assert len({c[-1] for c in calls}) == 7
    assert [r['job_id'] for r in workflow.read_yaml(path)['runs']] == [str(i) for i in range(101, 108)]


def test_partial_slurm_submission_keeps_manifest(workflow_configs, monkeypatch, tmp_path):
    path, batch = workflow.prepare_batch(_plan(workflow_configs, years='2018:2019'), 'slurm')
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, command, stderr='queue unavailable')
        return SimpleNamespace(stdout='123\n')

    monkeypatch.setattr(workflow.subprocess, 'run', submit)
    with pytest.raises(subprocess.CalledProcessError):
        workflow.submit_batch(path, batch, tmp_path / 'script.sh')
    manifest = workflow.read_yaml(path)
    assert manifest['runs'][0]['job_id'] == '123'
    assert manifest['runs'][1]['submission_error']


def test_removed_date_options_are_rejected(workflow_configs):
    path = workflow_configs['stacking']
    config = workflow.read_yaml(path)
    config['stacking']['t_length'] = 'all'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='were removed'):
        load_config(path, hemisphere='sh', sensors=['cryosat2'])


@pytest.mark.parametrize('sensors', [['unknown'], ['cryosat2', 'cryosat2'], ['icesat2', 'cryosat2']])
def test_invalid_mission_selections_are_rejected(workflow_configs, sensors):
    with pytest.raises(ValueError):
        workflow.stacking_configs(workflow_configs['stacking'], hemisphere='sh', sensors=sensors, years='2018')


class _InlinePool:
    def __init__(self, *args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def apply_async(self, function, args):
        return SimpleNamespace(get=lambda: function(*args))


@pytest.mark.parametrize('step', ['direction', 'merge'])
def test_stacking_worker_failures_reach_snapshot_status(workflow_configs, monkeypatch, step):
    from driftaware_sialt.stacking import pipeline
    configs = _plan(workflow_configs, start='2020-01-01', end='2020-01-02')
    configs[0]['stacking']['multiproc'] = True
    _, batch = workflow.prepare_batch(configs, 'local')
    monkeypatch.setattr(workflow, 'init_logger', lambda c: None)
    monkeypatch.setattr(pipeline, 'init_logger', lambda c: None)
    monkeypatch.setattr(pipeline.gridding_lib, 'define_grid', lambda *a, **kw: (object(), 1))
    monkeypatch.setattr(pipeline.mp, 'Pool', _InlinePool)

    def fail(*args):
        raise RuntimeError(f'{step} failed')

    monkeypatch.setattr(pipeline, 'process_direction', fail if step == 'direction' else lambda *a: None)
    monkeypatch.setattr(pipeline, 'merge_direction_stacks', fail)
    with pytest.raises(RuntimeError, match=f'{step} failed'):
        workflow.execute_snapshot(batch['runs'][0]['snapshot'], pipeline.run)
    assert workflow.read_yaml(Path(configs[0]['run']['directory'], 'status.yaml'))['state'] == 'failed'


@pytest.mark.parametrize('multiproc', [False, True])
def test_gridding_groups_by_filename_date_and_keeps_outputs_in_run(
    workflow_configs, monkeypatch, multiproc,
):
    from driftaware_sialt.gridding import gridding
    _, batch = _complete(workflow_configs, monkeypatch, years='2020')
    source = Path(batch['runs'][0]['directory'])
    (source / 'final.csv').unlink()
    for date in ('20200101', '20200102'):
        (source / f'ESACCI-SEAICE-L2P-DA-SITHICK-SIRAL-SH-{date}-fv2.0.csv').write_text('fixture')
    # A leftover directional file must not be mixed with final trajectories.
    (source / 'ESACCI-SEAICE-L2P-DA_F-SITHICK-SIRAL-SH-20200102-fv2.0.csv').write_text('partial')
    config = workflow.gridding_configs(workflow_configs['gridding'], source_run=source)[0]
    config['gridding']['multiproc'] = multiproc
    _, grid_batch = workflow.prepare_batch([config], 'local')
    monkeypatch.setattr(gridding, 'get_source_stack_metadata', lambda *a: {})
    monkeypatch.setattr(gridding.gridding_lib, 'define_grid', lambda *a, **kw: (object(), 1))
    monkeypatch.setattr(gridding, 'get_sea_ice_regions', lambda *a: (object(), {}))
    monkeypatch.setattr(gridding.mp, 'Pool', _InlinePool)
    grouped = []

    def process(cfg, files, *args):
        grouped.append(files)
        date = Path(files[0]).name.split('-')[-2]
        (Path(cfg['output_dir']['gridded_data']) / f'grid-{date}-fv2.0.nc').write_bytes(b'fixture')

    monkeypatch.setattr(gridding, 'process_file', process)
    workflow.execute_snapshot(grid_batch['runs'][0]['snapshot'], gridding.gridding)
    assert [len(files) for files in grouped] == [1, 1]
    directory = Path(config['run']['directory'])
    assert sorted(p.name for p in (directory / '2020/01').glob('*.nc')) == [
        'grid-20200101-fv2.0.nc', 'grid-20200102-fv2.0.nc']
    assert workflow.read_yaml(directory / 'status.yaml')['state'] == 'completed'


def test_unmerged_directional_outputs_do_not_count_as_success(workflow_configs, monkeypatch):
    configs = _plan(workflow_configs, years='2018')
    _, batch = workflow.prepare_batch(configs, 'local')
    monkeypatch.setattr(workflow, 'init_logger', lambda c: None)
    with pytest.raises(RuntimeError, match='no final'):
        workflow.execute_snapshot(batch['runs'][0]['snapshot'], lambda c: (
            Path(c['run']['directory']) / 'partial_F-20180101.csv').write_text('fixture'))
