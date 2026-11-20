import datetime as dt
from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import pytest
from shapely.geometry import box

import driftaware_sialt.stacking.pipeline as pipeline
from driftaware_sialt.config_loader import load_config
from driftaware_sialt.io_tools import read_dasit_csv
from driftaware_sialt.products.sea_ice_thickness import SeaIceThicknessMultiProducts
from driftaware_sialt.stacking.drift_aware_processor import grid_mission_fractions
from driftaware_sialt import cli, workflow


def _write_l2p(path, platforms=None, *, bad_metadata=False, legacy_coords=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(platforms) if platforms is not None else 3
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('time', n)
        values = {
            'latitude' if legacy_coords else 'lat': np.full(n, 75.0),
            'longitude' if legacy_coords else 'lon': np.full(n, 10.0),
            'radar_freeboard': np.full(n, 0.1),
            'sea_ice_freeboard': np.full(n, 0.2),
            'sea_ice_thickness': np.arange(n, dtype=float) + 1,
            'radar_freeboard_uncertainty': np.full(n, 0.01),
            'sea_ice_freeboard_uncertainty': np.full(n, 0.02),
            'sea_ice_thickness_uncertainty': np.full(n, 0.1),
            'snow_depth': np.full(n, 0.2),
            'time': np.arange(n, dtype=float),
        }
        for name, data in values.items():
            ds.createVariable(name, 'f8', ('time',))[:] = data
        ds.createVariable('flag_miz', 'i1', ('time',))[:] = np.zeros(n)
        if platforms is not None:
            flag = ds.createVariable('platform', 'i2', ('time',), fill_value=-999)
            flag[:] = platforms
            # Deliberately use non-standard codes to ensure metadata is honored.
            flag.flag_values = [13, 24, 35]
            if not bad_metadata:
                flag.flag_meanings = 'cryosat2 sentinel3a sentinel3b'


def _products(sensors, target='sea_ice_thickness'):
    return SeaIceThicknessMultiProducts(
        sensor=sensors, hem='nh', target_var=target, add_variable=['snow_depth'], out_epsg=6931)


def _read_day(products, directories, day=dt.datetime(2020, 4, 1)):
    products.get_file_list(directories)
    products.get_file_dates()
    products.get_target_files(day, day + dt.timedelta(days=1))
    products.get_product([s for s in products.sensor if products.target_files[s]])
    return products.product


def _shared_config(files):
    config = workflow.read_yaml(files['common'])
    config['input_dir'] = {'multi-mission': 'input/{hemisphere}/multi-mission'}
    workflow._write_yaml(files['common'], config)
    return workflow.stacking_configs(files['stacking'], hemisphere='nh',
                                     sensors=['multi-mission'], years='2020')[0]


def test_all_platforms_selection_tracks_daily_missions(workflow_configs, monkeypatch):
    config = _shared_config(workflow_configs)
    root = workflow_configs['common'].parent / 'input/nh/multi-mission'
    first = root / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200401-fv5p0.nc'
    second = root / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200402-fv5p0.nc'
    _write_l2p(first, [1, 2, 13, 24, 35])
    with netCDF4.Dataset(first, 'a') as ds:
        ds['platform'].flag_values = [1, 2, 13, 24, 35]
        ds['platform'].flag_meanings = 'ers2 envisat cryosat2 sentinel3a sentinel3b'
    _write_l2p(second, [24, 24])
    products = _products(config['options']['sensor'])
    reads = []
    original = products._read_l2p

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(products, '_read_l2p', read)
    data = _read_day(products, config['input_dir'])
    assert reads == [str(first)]
    assert data.sensor.value_counts().to_dict() == {
        'ers2': 1, 'envisat': 1, 'cryosat2': 1, 'sentinel3a': 1, 'sentinel3b': 1}
    products.get_target_files(dt.datetime(2020, 4, 2), dt.datetime(2020, 4, 3))
    products.get_product(products.sensor)
    assert reads == [str(first), str(second)]
    assert products.product.sensor.tolist() == ['sentinel3a', 'sentinel3a']
    for sensor in ('ers2', 'envisat', 'cryosat2', 'sentinel3b'):
        assert products.product[sensor].sum() == 0


def test_multi_mission_selection_survives_snapshots_and_gridding(workflow_configs, monkeypatch):
    config = _shared_config(workflow_configs)
    assert config['options']['sensor_selection'] == 'multi-mission'
    assert 'icesat2' not in config['options']['sensor']
    assert config['output_dir']['trajectories'].endswith('/multi-mission')
    _, batch = workflow.prepare_batch([config], 'local')
    source = batch['runs'][0]
    monkeypatch.setattr(workflow, 'init_logger', lambda c: None)
    workflow.execute_snapshot(source['snapshot'], lambda c: (
        Path(c['run']['directory']) / 'final.csv').write_text('fixture'))
    # Gridding does not require the original input repository to be configured.
    common = workflow.read_yaml(workflow_configs['common'])
    common['input_dir'] = {}
    workflow._write_yaml(workflow_configs['common'], common)
    grid, = workflow.gridding_configs(workflow_configs['gridding'], source_run=source['directory'])
    assert grid['options']['sensor'] == config['options']['sensor']
    assert grid['options']['sensor_selection'] == 'multi-mission'
    assert grid['output_dir']['gridded_data'].endswith('/multi-mission')
    _, grid_batch = workflow.prepare_batch([grid], 'slurm')
    assert workflow.read_snapshot(grid_batch['runs'][0]['snapshot']) == grid


@pytest.mark.parametrize('sensors', [['multi-mission', 'cryosat2'],
                                   ['multi-mission', 'icesat2'], ['multi-mission', 'multi-mission']])
def test_multi_mission_cannot_be_combined_with_other_selections(workflow_configs, sensors):
    with pytest.raises(ValueError, match='selected alone'):
        workflow.stacking_configs(workflow_configs['stacking'], hemisphere='nh',
                                 sensors=sensors, years='2020')


def test_multi_mission_requires_shared_input(workflow_configs):
    with pytest.raises(ValueError, match='requires input_dir.multi-mission'):
        workflow.stacking_configs(workflow_configs['stacking'], hemisphere='nh',
                                 sensors=['multi-mission'], years='2020')


def test_multi_mission_cli_previews_without_creating_runs(workflow_configs, tmp_path, capsys):
    _shared_config(workflow_configs)
    before = set(tmp_path.rglob('*'))
    assert cli.main(['stacking', str(workflow_configs['stacking']), '--hemisphere', 'nh',
                     '--sensors', 'multi-mission', '--years', '2020', '--dry-run']) == 0
    assert before == set(tmp_path.rglob('*'))
    assert '/trajectories/multi-mission/' in capsys.readouterr().out


@pytest.mark.parametrize('target', ['sea_ice_thickness', 'sea_ice_freeboard', 'radar_freeboard'])
@pytest.mark.parametrize('sensors', [
    ['cryosat2'], ['sentinel3a', 'sentinel3b'], ['cryosat2', 'sentinel3a', 'sentinel3b'],
])
def test_shared_file_selects_missions_and_is_read_once(tmp_path, monkeypatch, sensors, target):
    filename = tmp_path / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200401-fv5p0.nc'
    _write_l2p(filename, [13, 24, 35, 13, 24, 35])
    products = _products(sensors, target)
    reads = []
    original = products._read_l2p

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(products, '_read_l2p', read)
    result = _read_day(products, {'multi-mission': str(tmp_path)})

    assert reads == [str(filename)]
    assert result.sensor.value_counts().to_dict() == {s: 2 for s in sensors}
    assert len(result) == 2 * len(sensors)
    assert set(result.platform) == set(sensors)
    assert result.crs.to_epsg() == 6931
    for sensor in sensors:
        assert (result[sensor] == (result.sensor == sensor).astype(int)).all()


def test_missing_mission_has_zero_counts_and_no_previous_day_observations(tmp_path):
    _write_l2p(tmp_path / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200401-fv5p0.nc', [13, 24, 35])
    _write_l2p(tmp_path / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200402-fv5p0.nc', [24, 24])
    sensors = ['cryosat2', 'sentinel3a', 'sentinel3b']
    products = _products(sensors)
    _read_day(products, {'multi-mission': str(tmp_path)})
    products.get_target_files(dt.datetime(2020, 4, 2), dt.datetime(2020, 4, 3))
    products.get_product(sensors)
    data = products.product

    assert data.sensor.tolist() == ['sentinel3a', 'sentinel3a']
    assert data.cryosat2.sum() == data.sentinel3b.sum() == 0
    x, y = data.geometry.iloc[0].coords[0]
    grid = gpd.GeoDataFrame(geometry=[box(x - 1, y - 1, x + 1, y + 1)], crs=data.crs)
    result = grid_mission_fractions(data, grid, sensors, list(data.sensor.unique()))
    assert result.sentinel3a_sum.iloc[0] == 2
    assert result.cryosat2_sum.iloc[0] == result.sentinel3b_sum.iloc[0] == 0


def test_selected_mission_can_be_entirely_absent(tmp_path):
    _write_l2p(tmp_path / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200401-fv5p0.nc', [24, 35])
    assert _read_day(_products(['cryosat2']), {'multi-mission': str(tmp_path)}).empty


def test_platform_selection_preserves_qc_and_excludes_unknown_or_masked_flags(tmp_path):
    path = tmp_path / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200401-fv5p0.nc'
    _write_l2p(path, [13, 13, 24, 35, 99, -999])
    with netCDF4.Dataset(path, 'a') as ds:
        ds['flag_miz'][1] = 2
        ds['sea_ice_thickness'][2] = np.nan
    data = _read_day(_products(['cryosat2', 'sentinel3a', 'sentinel3b']), {'multi-mission': str(tmp_path)})
    assert data.sensor.tolist() == ['cryosat2', 'sentinel3b']
    assert data.sea_ice_thickness.tolist() == [1, 4]


@pytest.mark.parametrize('bad_metadata', [False, True])
def test_shared_files_require_platform_metadata(tmp_path, bad_metadata):
    path = tmp_path / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200401-fv5p0.nc'
    _write_l2p(path, [13] if bad_metadata else None, bad_metadata=bad_metadata)
    with pytest.raises(ValueError, match='platform'):
        _read_day(_products(['cryosat2']), {'multi-mission': str(tmp_path)})


@pytest.mark.parametrize('legacy_coords', [False, True])
def test_legacy_separate_files_remain_supported(tmp_path, legacy_coords):
    directories = {}
    for sensor in ['cryosat2', 'sentinel3a']:
        root = tmp_path / sensor
        _write_l2p(root / f'ESACCI-SEAICE-L2P-SITHICK-{sensor}-NH-20200401-fv4p0.nc', legacy_coords=legacy_coords)
        directories[sensor] = str(root)
    data = _read_day(_products(list(directories)), directories)
    assert data.sensor.value_counts().to_dict() == {'cryosat2': 3, 'sentinel3a': 3}


def test_platform_files_in_mission_paths_are_filtered_without_duplication(tmp_path):
    _write_l2p(tmp_path / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200401-fv5p0.nc', [13, 24, 35])
    data = _read_day(_products(['cryosat2', 'sentinel3a']), {
        'cryosat2': str(tmp_path), 'sentinel3a': str(tmp_path)})
    assert data.sensor.tolist() == ['cryosat2', 'sentinel3a']


def test_shared_source_takes_precedence_over_legacy_directories(tmp_path):
    shared, legacy = tmp_path / 'shared', tmp_path / 'legacy'
    _write_l2p(shared / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200401-fv5p0.nc', [13, 24])
    _write_l2p(legacy / 'ESACCI-SEAICE-L2P-SITHICK-cryosat2-NH-20200401-fv4p0.nc')
    data = _read_day(_products(['cryosat2']), {
        'multi-mission': str(shared), 'cryosat2': str(legacy)})
    assert data.sensor.tolist() == ['cryosat2']


def test_multiple_daily_files_are_combined_and_same_path_is_read_once(tmp_path):
    a = tmp_path / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200401-a.nc'
    b = tmp_path / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200401-b.nc'
    _write_l2p(a, [13])
    _write_l2p(b, [24])
    products = _products(['cryosat2', 'sentinel3a'])
    products.target_files.update(cryosat2=[str(a), str(a), str(b)], sentinel3a=[str(a), str(b)])
    products.get_product(products.sensor)
    assert products.product.sensor.tolist() == ['cryosat2', 'sentinel3a']


def test_shared_directory_resolves_config_placeholders(tmp_path):
    config_file = tmp_path / 'stacking.yaml'
    config_file.write_text(
        f'base_dir: {tmp_path}\nversion: 1.1\nstage: stacking\n'
        'options: {sensor: [cryosat2, sentinel3a], hemisphere: sh}\n'
        'input_dir: {multi-mission: "input/{hemisphere}/multi-mission"}\n'
        'output_dir: {}\nauxiliary: {}\nlogging: logs\n'
        'stacking:\n  parallel: {enabled: false, workers: 1}\n'
        '  climatology: {sit: {}, tfb: {}}\n  thermo_change: {enabled: false}\n')
    config = load_config(config_file)
    assert config['input_dir']['multi-mission'] == str(tmp_path / 'input/sh/multi-mission')
    assert config['options']['sensor'] == ['cryosat2', 'sentinel3a']


@pytest.mark.parametrize('multi_mission', [False, True])
def test_stacking_handles_a_day_without_any_selected_platform(
    tmp_path, monkeypatch, workflow_configs, multi_mission,
):
    root = tmp_path / 'shared'
    _write_l2p(root / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200401-fv5p0.nc', [24, 35])
    _write_l2p(root / 'ESACCI-SEAICE-L2P-SITHICK-NH-20200402-fv5p0.nc', [13, 13])
    output = tmp_path / 'output'
    output.mkdir()
    point = gpd.GeoSeries.from_xy([10], [75], crs=4326).to_crs(6931).iloc[0]
    grid = gpd.GeoDataFrame(geometry=[box(point.x - 1, point.y - 1, point.x + 1, point.y + 1)], crs=6931)
    config = {
        'version': '1.1',
        'options': {
            'sensor': ['cryosat2'], 'target_variable': 'sea_ice_thickness',
            'add_variable': ['snow_depth'], 'hemisphere': 'nh', 'out_epsg': 'EPSG:6931',
            'ice_conc_products': ['concentration'], 'ice_drift_products': ['drift'],
        },
        'input_dir': {'multi-mission': str(root)},
        'output_dir': {'trajectories': str(output)},
        'auxiliary': {'ice_conc': {}, 'ice_drift': {}},
        'run': {'output_start': '2020-04-01', 'output_end': '2020-04-02',
                'processing_start': '2020-04-01', 'processing_end': '2020-04-02'},
        'stacking': {
            't_window': 1,
            'mode': 'f', 'filename_prefix': 'ESACCI-SEAICE',
            'hist': {'n_bins': 10, 'range': {'thickness': [0, 10]}},
            'thermo_change': {'enabled': False},
        },
    }
    if multi_mission:
        planned = _shared_config(workflow_configs)
        config['options']['sensor'] = planned['options']['sensor']

    class Concentration:
        product_id = 'concentration'
        target_files = 'concentration.nc'

        def get_ice_concentration(self, file):
            return object()

        def interp_ice_concentration(self, data, x, y):
            return np.ones(len(x))

    class Drift:
        target_files = 'drift.nc'

        def get_ice_drift(self, file, concentration):
            pass

    monkeypatch.setattr(pipeline, 'init_logger', lambda config: None)
    monkeypatch.setattr(pipeline.SeaIceConcentrationProducts, 'load_products', lambda *a, **kw: [Concentration()])
    monkeypatch.setattr(pipeline.SeaIceDriftProducts, 'load_products', lambda *a, **kw: [Drift()])
    monkeypatch.setattr(pipeline, 'select_aux_product', lambda products, *args: products[0])

    pipeline.process_direction(config, 'f', grid)

    files = list(output.glob('*.csv'))
    assert len(files) == (2 if multi_mission else 1)
    second, = [path for path in files if '20200402' in path.name]
    result = read_dasit_csv(second)
    assert result.cryosat2_cnt.tolist() == [2]
    if multi_mission:
        first, = [path for path in files if '20200401' in path.name]
        data = read_dasit_csv(first)
        assert data.sentinel3a_cnt.tolist() == data.sentinel3b_cnt.tolist() == [1]
        assert data.cryosat2_cnt.tolist() == data.ers2_cnt.tolist() == data.envisat_cnt.tolist() == [0]
        assert result.sentinel3a_cnt.tolist() == result.sentinel3b_cnt.tolist() == [0]
