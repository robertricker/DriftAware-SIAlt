from pathlib import Path

import pytest

import driftaware_sialt.cli as cli
import driftaware_sialt.input_data as input_data
from driftaware_sialt import workflow


def _args(config, *extra):
    return [str(config), '--hemisphere', 'sh', '--sensors', 'cryosat2', *extra]


def test_syncdata_command_only_synchronizes_inputs(workflow_configs, monkeypatch):
    synchronized = []
    monkeypatch.setattr(cli, 'init_logger', lambda config: None)
    monkeypatch.setattr(input_data, 'sync_required_input_data', lambda c: synchronized.append(c))
    monkeypatch.setattr(cli, 'run_stage', lambda c: pytest.fail('syncdata must not process'))
    assert cli.main(['syncdata', *_args(workflow_configs['stacking'], '--years', '2018:2019')]) == 0
    assert len(synchronized) == 2
    assert synchronized[0]['run']['processing_start'] == '2017-12-28'


def test_local_years_run_sequentially_without_downloading(workflow_configs, monkeypatch):
    processed = []
    monkeypatch.setattr(workflow, 'init_logger', lambda config: None)
    monkeypatch.setattr(input_data, 'sync_required_input_data', lambda c: pytest.fail('unexpected download'))

    def stage(config):
        processed.append(config['run']['output_start'])
        (Path(config['run']['directory']) / 'result.csv').write_text('fixture')

    monkeypatch.setattr(cli, 'run_stage', stage)
    assert cli.main(['stacking', *_args(workflow_configs['stacking'], '--years', '2018:2020')]) == 0
    assert processed == ['2018-01-01', '2019-01-01', '2020-01-01']


def test_dry_run_has_no_writes_or_execution(workflow_configs, monkeypatch, tmp_path, capsys):
    before = set(tmp_path.rglob('*'))
    monkeypatch.setattr(cli, 'run_stage', lambda c: pytest.fail('unexpected processing'))
    monkeypatch.setattr(workflow, 'submit_batch', lambda *a: pytest.fail('unexpected submission'))
    assert cli.main(['stacking', *_args(workflow_configs['stacking'], '--years', '2018:2024', '--dry-run')]) == 0
    assert before == set(tmp_path.rglob('*'))
    output = capsys.readouterr().out
    assert 'processing 2017-12-28..2019-01-04' in output
    assert output.count('outputs ') == 7


def test_stage_command_rejects_mismatched_configuration(workflow_configs):
    with pytest.raises(SystemExit):
        cli.main(['stacking', *_args(workflow_configs['gridding'], '--years', '2018')])


@pytest.mark.parametrize('extra', [
    [], ['--years', '2018', '--start', '2018-01-01', '--end', '2018-12-31'],
    ['--start', '2018-01-01'], ['--end', '2018-12-31'],
])
def test_missing_or_conflicting_dates_are_rejected(workflow_configs, extra):
    with pytest.raises(SystemExit):
        cli.main(['stacking', *_args(workflow_configs['stacking'], *extra)])


def test_submit_dry_run_does_not_submit_or_write(workflow_configs, tmp_path, monkeypatch):
    script = tmp_path / 'script.sh'
    script.write_text('srun driftaware stacking "$1"\n')
    before = set(tmp_path.rglob('*'))
    monkeypatch.setattr(workflow, 'submit_batch', lambda *a: pytest.fail('unexpected submission'))
    assert cli.main(['submit', 'stacking', *_args(
        workflow_configs['stacking'], '--years', '2018', '--slurm-script', str(script), '--dry-run')]) == 0
    assert before == set(tmp_path.rglob('*'))


def test_submit_requires_slurm_before_creating_output_dirs(workflow_configs, tmp_path, monkeypatch):
    script = tmp_path / 'script.sh'
    script.write_text('srun driftaware stacking "$1"\n')
    before = set(tmp_path.rglob('*'))
    monkeypatch.setattr(cli.shutil, 'which', lambda name: None)
    with pytest.raises(SystemExit):
        cli.main(['submit', 'stacking', *_args(
            workflow_configs['stacking'], '--years', '2018', '--slurm-script', str(script))])
    assert before == set(tmp_path.rglob('*'))


def test_submit_command_freezes_and_submits_every_year(workflow_configs, tmp_path, monkeypatch):
    script = tmp_path / 'script.sh'
    script.write_text('srun driftaware stacking "$1"\n')
    monkeypatch.setattr(cli.shutil, 'which', lambda name: '/usr/bin/sbatch')
    submitted = []

    def submit(path, manifest, script_path):
        assert Path(path).is_file()
        for entry in manifest['runs']:
            assert workflow.read_snapshot(entry['snapshot'])['stage'] == 'stacking'
            submitted.append(entry)

    monkeypatch.setattr(workflow, 'submit_batch', submit)
    assert cli.main(['submit', 'stacking', *_args(
        workflow_configs['stacking'], '--years', '2018:2024', '--slurm-script', str(script))]) == 0
    assert len(submitted) == 7


def test_gridding_cli_uses_completed_source_batch(workflow_configs, monkeypatch, tmp_path):
    monkeypatch.setattr(workflow, 'init_logger', lambda c: None)

    def stage(config):
        suffix = 'csv' if config['stage'] == 'stacking' else 'nc'
        (Path(config['run']['directory']) / f'final.{suffix}').write_text('fixture')

    monkeypatch.setattr(cli, 'run_stage', stage)
    cli.main(['stacking', *_args(workflow_configs['stacking'], '--years', '2018:2019')])
    manifest, = tmp_path.glob('output_data/v2.0/trajectories/cryosat2/batches/*.yaml')
    assert cli.main(['gridding', str(workflow_configs['gridding']), '--source-batch', str(manifest)]) == 0
    grid_manifest, = tmp_path.glob('output_data/v2.0/l3c/cryosat2/batches/*.yaml')
    assert len(workflow.read_yaml(grid_manifest)['runs']) == 2


def test_worker_executes_only_its_snapshot(workflow_configs, monkeypatch):
    config = workflow.stacking_configs(workflow_configs['stacking'], hemisphere='sh',
                                      sensors=['cryosat2'], years='2018')[0]
    _, batch = workflow.prepare_batch([config], 'slurm')
    snapshot = batch['runs'][0]['snapshot']
    monkeypatch.setattr(workflow, 'init_logger', lambda c: None)
    monkeypatch.setattr(cli, 'run_stage', lambda c: (
        Path(c['run']['directory']) / 'result.csv').write_text('fixture'))
    with pytest.raises(SystemExit):
        cli.main(['stacking', snapshot, '--hemisphere', 'nh'])
    with pytest.raises(ValueError, match='gridding snapshot'):
        cli.main(['gridding', snapshot])
    assert cli.main(['stacking', snapshot]) == 0


@pytest.mark.parametrize('arguments', [
    ['sync', 'stacking.yaml'], ['run', 'stacking.yaml'], ['stacking.yaml'],
])
def test_removed_commands_are_rejected(arguments):
    with pytest.raises(SystemExit, match='2'):
        cli.main(arguments)
