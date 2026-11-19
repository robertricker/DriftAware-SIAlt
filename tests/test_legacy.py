import geopandas as gpd
import pytest
from shapely.geometry import MultiPoint

from driftaware_sialt import cli, legacy, workflow
from driftaware_sialt.io_tools import write_dasit_csv


def _run(root, hemisphere='nh', dates=('20181001', '20181002')):
    directory = root / f'sea_ice_thickness-{hemisphere}-16fr-epsg693{1 if hemisphere == "nh" else 2}_141-old'
    directory.mkdir(parents=True)
    config = {
        'options': {'out_epsg': f'EPSG:693{1 if hemisphere == "nh" else 2}',
                    'target_variable': 'sea_ice_thickness'},
        'stacking': {'mode': 'fr', 't_window': 16,
                     'hist': {'n_bins': 40, 'range': {'freeboard': [0, 1], 'thickness': [0, 10]}}},
    }
    data = gpd.GeoDataFrame({
        'sea_ice_thickness': [1.0], 'sea_ice_thickness_hist': ['1 ' * 40],
        'dt_days': [0], 'snow_depth': [0.2],
        'cryosat2_cnt': [1], 'sentinel3a_cnt': [0], 'sentinel3b_cnt': [0],
    }, geometry=[MultiPoint([(0, 0), (1, 1)])], crs=config['options']['out_epsg'])
    for date in dates:
        file = directory / ('ESACCI-SEAICE-L2P-DA-SITHICK-'
                            'SIRAL_CRYOSAT2_SRAL_SENTINEL3A_SRAL_SENTINEL3B-'
                            f'{hemisphere.upper()}-{date}-fv2.0.csv')
        write_dasit_csv(data, file, config)
    return directory


def test_registration_reconstructs_metadata_without_altering_csvs(tmp_path, workflow_configs):
    directory = _run(tmp_path / 'archives')
    original = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.glob('*.csv')}
    manifests = legacy.register_trajectories(directory, completed=True)
    config = workflow.read_snapshot(directory / workflow.SNAPSHOT_NAME)
    assert config['options']['sensor'] == ['cryosat2', 'sentinel3a', 'sentinel3b']
    assert config['options']['out_epsg'] == 'EPSG:6931'
    assert config['stacking']['t_window'] == 16
    assert config['stacking']['hist']['n_bins'] == 40
    assert config['software_git_commit'] is None
    assert config['run']['output_start'] == '2018-10-01'
    assert config['run']['processing_start'] is config['run']['processing_end'] is None
    assert config['legacy_import']['csv_count'] == 2
    status = workflow.read_yaml(directory / 'status.yaml')
    assert status['state'] == 'completed' and status['run_id'] == config['run']['id']
    assert 'not a recovered SLURM status' in status['completion_basis']
    assert original == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.glob('*.csv')}
    assert len(workflow.gridding_configs(workflow_configs['gridding'], source_batch=manifests[0])) == 1


def test_registration_groups_hemispheres_and_skips_empty_runs(tmp_path):
    north = _run(tmp_path)
    south = _run(tmp_path, 'sh', ('20200101', '20200102'))
    empty = tmp_path / 'empty'
    empty.mkdir()
    paths = legacy.register_trajectories(tmp_path, completed=True)
    assert len(paths) == 2
    assert {workflow.read_yaml(p)['id'].split('-')[1] for p in paths} == {'nh', 'sh'}
    assert not (empty / workflow.SNAPSHOT_NAME).exists()
    assert all((p / 'status.yaml').exists() for p in (north, south))


def test_registration_is_idempotent_and_dry_run_does_not_write(tmp_path):
    directory = _run(tmp_path)
    before = {p.name for p in directory.iterdir()}
    assert legacy.register_trajectories(directory, dry_run=True) == []
    assert before == {p.name for p in directory.iterdir()}
    with pytest.raises(ValueError, match='--completed'):
        legacy.register_trajectories(directory)
    legacy.register_trajectories(directory, completed=True)
    snapshot = (directory / workflow.SNAPSHOT_NAME).read_bytes()
    status = (directory / 'status.yaml').read_bytes()
    legacy.register_trajectories(directory, completed=True)
    assert (directory / workflow.SNAPSHOT_NAME).read_bytes() == snapshot
    assert (directory / 'status.yaml').read_bytes() == status


def test_modified_legacy_csvs_are_rejected_for_gridding(tmp_path):
    directory = _run(tmp_path)
    legacy.register_trajectories(directory, completed=True)
    path = next(directory.glob('*.csv'))
    with path.open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='changed after registration'):
        workflow.source_snapshots(source_run=directory)


def test_registration_accepts_large_geometry_cells(tmp_path):
    directory = _run(tmp_path, dates=('20181001',))
    path = next(directory.glob('*.csv'))
    text = path.read_text().replace(MultiPoint([(0, 0), (1, 1)]).wkt,
                                    'MULTIPOINT (' + ', '.join(['(0 0)'] * 30000) + ')')
    assert len(text) > 131072
    path.write_text(text)
    assert legacy.inspect_legacy(directory)['legacy_import']['csv_count'] == 1


@pytest.mark.parametrize('corruption', ['inconsistent', 'directional', 'crs_only', 'empty', 'partial_marker'])
def test_registration_rejects_incomplete_or_inconsistent_runs(tmp_path, corruption):
    directory = _run(tmp_path)
    files = sorted(directory.glob('*.csv'))
    if corruption == 'inconsistent':
        text = files[1].read_text().replace('"window_days":16', '"window_days":15')
        files[1].write_text(text)
    elif corruption == 'directional':
        files[1].rename(files[1].with_name(files[1].name.replace('-DA-', '-DA_F-')))
    elif corruption == 'crs_only':
        lines = files[1].read_text().splitlines(keepends=True)
        files[1].write_text('# EPSG:6931\n' + ''.join(lines[1:]))
    elif corruption == 'empty':
        lines = files[1].read_text().splitlines(keepends=True)
        files[1].write_text(''.join(lines[:2]))
    else:
        (directory / 'status.yaml').write_text('state: completed\n')
    with pytest.raises(ValueError):
        legacy.register_trajectories(directory, completed=True)
    assert not (directory / workflow.SNAPSHOT_NAME).exists()


def test_registered_sources_cannot_rerun_stacking(tmp_path):
    directory = _run(tmp_path)
    legacy.register_trajectories(directory, completed=True)
    snapshot = directory / workflow.SNAPSHOT_NAME
    with pytest.raises(ValueError, match='cannot rerun stacking'):
        workflow.execute_snapshot(snapshot, lambda c: pytest.fail('must not execute'))
    with pytest.raises(SystemExit):
        cli.main(['stacking', str(snapshot), '--dry-run'])
    assert not (directory / 'execution.lock').exists()


def test_registration_command_accepts_a_parent_directory(tmp_path):
    directory = _run(tmp_path)
    assert cli.main(['register-trajectories', str(tmp_path), '--dry-run']) == 0
    assert not (directory / 'status.yaml').exists()
    assert cli.main(['register-trajectories', str(tmp_path), '--completed']) == 0
    assert (directory / 'status.yaml').exists()
