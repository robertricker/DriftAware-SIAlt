"""Analytical flow and native-file regressions for advection and strain rates."""

import datetime as dt
from types import SimpleNamespace

import netCDF4
import numpy as np
import pandas as pd
from pyproj import Transformer
import pytest
from shapely.geometry import Point

from driftaware_sialt.products.drift_timing import EPOCH
from driftaware_sialt.products.sea_ice_drift import SeaIceDriftProducts, select_drift_product
from driftaware_sialt.stacking.drift_aware_processor import DriftAwareProcessor
from driftaware_sialt import workflow
from driftaware_sialt.io_tools import read_dasit_csv
import driftaware_sialt.stacking.pipeline as pipeline
import driftaware_sialt.input_data as input_data


DAY = dt.timedelta(days=1)
NOON = dt.datetime(2020, 12, 31, 12)


def _file(tmp_path, product, end=NOON, *, hemisphere='nh', hours=None,
          vector_hours=None, units='days since 2001-01-01', uncertainty_units='km'):
    hours = hours if hours is not None else (48 if product == 'osi405' else 24)
    start = end - dt.timedelta(hours=hours)
    vector_hours = hours if vector_hours is None else vector_hours
    name = (f'ice_drift_{hemisphere}_test_{start:%Y%m%d%H%M}-{end:%Y%m%d%H%M}.nc'
            if product == 'osi405' else f'ice_drift_{hemisphere}_test_{end:%Y%m%d%H%M}.nc')
    directory = tmp_path / product
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    x, y = np.meshgrid(np.array([-75000., -25000., 25000., 75000.]),
                       np.array([75000., 25000., -25000., -75000.]))
    crs = 6931 if hemisphere == 'nh' else 6932
    project = Transformer.from_crs(crs, 4326, always_xy=True)
    lon, lat = project.transform(x, y)
    lon1, lat1 = project.transform(x + 100 * vector_hours, y - 50 * vector_hours)
    with netCDF4.Dataset(path, 'w') as ds:
        for dim, size in (('time', 1), ('yc', 4), ('xc', 4), ('nv', 2)):
            ds.createDimension(dim, size)
        time = ds.createVariable('time', 'f8', ('time',))
        time.units = units
        time[:] = netCDF4.date2num([end], units)
        bounds = ds.createVariable('time_bnds', 'f8', ('time', 'nv'))
        # Exercise CF units inherited from the time coordinate.
        bounds[:] = netCDF4.date2num([start, end], units)
        for key, values in (('lon', lon), ('lat', lat), ('lon1', lon1), ('lat1', lat1)):
            ds.createVariable(key, 'f8', ('yc', 'xc'))[:] = values
        unc = ds.createVariable('uncert_dX_and_dY', 'f8', ('time', 'yc', 'xc'))
        unc.units = uncertainty_units
        unc[:] = 10 * vector_hours / (1000 if uncertainty_units == 'km' else 1)
        ds.createVariable('status_flag', 'i2', ('time', 'yc', 'xc'))[:] = 0
        if product == 'osi405':
            for key, seconds in (('dt0', 0), ('dt1', (vector_hours - hours) * 3600)):
                offset = ds.createVariable(key, 'f8', ('time', 'yc', 'xc'))
                offset.units = 'seconds'
                offset[:] = seconds
        else:
            for key, date in (('t0', start), ('t1', start + dt.timedelta(hours=vector_hours))):
                clock = ds.createVariable(key, 'f8', ('time', 'yc', 'xc'))
                clock.units = 'hours since 1990-01-01'
                clock[:] = netCDF4.date2num(date, clock.units)
    return path


def _product(product, paths=(), hemisphere='nh'):
    result = SeaIceDriftProducts(product_id=product, hem=hemisphere,
                                out_epsg=f'EPSG:{6931 if hemisphere == "nh" else 6932}')
    result.file_list = list(map(str, paths))
    result.get_file_dates()
    return result


@pytest.mark.parametrize('product', ['osi455', 'osi435', 'osi405'])
@pytest.mark.parametrize('hemisphere', ['nh', 'sh'])
@pytest.mark.parametrize('uncertainty_units', ['km', 'm'])
def test_native_intervals_and_units_produce_the_same_velocity(
    tmp_path, product, hemisphere, uncertainty_units,
):
    path = _file(tmp_path, product, hemisphere=hemisphere, uncertainty_units=uncertainty_units)
    drift = _product(product, [path], hemisphere)
    x, y = np.meshgrid([-25000., 0., 25000.], [25000., 0., -25000.])
    drift.get_ice_drift(path, {'xc': x, 'yc': y, 'ice_conc': np.ones_like(x) * 100})
    u, v, sigma = drift.drift_correction(x.ravel(), y.ravel())
    np.testing.assert_allclose(u, 100, atol=1e-7)
    np.testing.assert_allclose(v, -50, atol=1e-7)
    np.testing.assert_allclose(sigma, 10, atol=1e-7)
    start, end = drift.time_interval(str(path))
    np.testing.assert_allclose(drift.ice_drift['time_bnds'],
                               [(start - EPOCH).total_seconds(), (end - EPOCH).total_seconds()])
    assert (end - start).total_seconds() == (48 if product == 'osi405' else 24) * 3600


@pytest.mark.parametrize('product', ['osi455', 'osi435', 'osi405'])
def test_per_vector_durations_are_used_before_interpolation(tmp_path, product):
    path = _file(tmp_path, product, vector_hours=12)
    drift = _product(product, [path])
    x, y = np.meshgrid([-25000., 25000.], [25000., -25000.])
    drift.get_ice_drift(path, {'xc': x, 'yc': y, 'ice_conc': np.ones_like(x) * 100})
    u, v, sigma = drift.drift_correction([0.], [0.])
    np.testing.assert_allclose([u[0], v[0], sigma[0]], [100, -50, 10], atol=1e-7)


@pytest.mark.parametrize('x_reverse', [False, True])
@pytest.mark.parametrize('y_reverse', [False, True])
@pytest.mark.parametrize('flow,expected', [
    ('translation', (0., 0.)), ('rotation', (0., 0.)),
    ('expansion', (.02, 0.)), ('convergence', (-.02, 0.)),
    ('y_expansion', (.01, .01)), ('shear', (0., .01)),
])
def test_strain_invariants_for_known_flows_include_grid_edges(x_reverse, y_reverse, flow, expected):
    x = np.arange(-3, 4, dtype=float) * 25000
    y = np.arange(-3, 4, dtype=float) * 50000  # Distinct x/y spacing.
    x, y = x[::-1] if x_reverse else x, y[::-1] if y_reverse else y
    xx, yy = np.meshgrid(x, y)
    zero = np.zeros_like(xx)
    rate = .01 / 24  # Convert a per-day strain rate to velocity in m/hour.
    fields = {
        'translation': (zero + 100, zero - 50), 'rotation': (-rate * yy, rate * xx),
        'expansion': (rate * xx, rate * yy), 'convergence': (-rate * xx, -rate * yy),
        'y_expansion': (zero, rate * yy), 'shear': (rate * yy, zero),
    }
    drift = _product('osi455')
    drift.ice_drift = {'xc': xx, 'yc': yy, 'u': fields[flow][0], 'v': fields[flow][1],
                       'velocity_unc': zero}
    div, shear = drift.deformation(xx.ravel(), yy.ravel())
    np.testing.assert_allclose(div, expected[0], atol=1e-8)
    np.testing.assert_allclose(shear, expected[1], atol=1e-8)


def _parcels(hour):
    clock = (NOON.replace(hour=hour) - EPOCH).total_seconds()
    return pd.DataFrame({
        'xu': [0.], 'yu': [0.], 'geometry': [[Point(0, 0)]], 't0': [clock],
        'time_current': [clock], 'dt_days': [0], 'sea_ice_thickness_drift_unc': [0.],
        'divergence': [[0.]], 'shear': [[0.]], '_ice_conc_products': ['sic'], '_ice_drift_products': [''],
    })


def _processor():
    return DriftAwareProcessor(SimpleNamespace(sensor=['cryosat2'], target_var='sea_ice_thickness',
                                               add_variable=[], out_epsg='EPSG:6931'))


SIC = SimpleNamespace(product_id='sic', ice_conc_ahead=None,
                      interp_ice_concentration=lambda ds, x, y: np.ones(len(x)))


@pytest.mark.parametrize('product', ['osi455', 'osi435', 'osi405'])
@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('hour', [0, 6, 12, 23])
@pytest.mark.parametrize('fallback', [-3, 0, 3])
def test_advection_clock_is_independent_of_product_and_fallback(
    tmp_path, product, direction, hour, fallback,
):
    path = _file(tmp_path, product, end=NOON + dt.timedelta(days=fallback))
    drift = _product(product, [path])
    x, y = np.meshgrid([-25000., 25000.], [25000., -25000.])
    drift.get_ice_drift(path, {'xc': x, 'yc': y, 'ice_conc': x * 0 + 100})
    frame = _parcels(hour)
    source_time = frame.time_current.iloc[0]
    target = NOON + direction * DAY
    result = _processor().apply_drift_correction(
        1, frame, drift, SIC, None, 0., direction, target_time=(target - EPOCH).total_seconds())
    elapsed = ((target - EPOCH).total_seconds() - source_time) / 3600
    np.testing.assert_allclose(result.xu, 100 * elapsed, atol=1e-6)
    np.testing.assert_allclose(result.yu, -50 * elapsed, atol=1e-6)
    np.testing.assert_allclose(result.sea_ice_thickness_drift_unc, (10 * elapsed) ** 2)
    assert result.time_current.iloc[0] == (target - EPOCH).total_seconds()


def test_product_switch_and_forward_reverse_round_trip(tmp_path):
    processor = _processor()
    frame = _parcels(12)
    for product, direction, target in (
            ('osi455', 1, NOON + DAY), ('osi405', 1, NOON + 2 * DAY),
            ('osi435', -1, NOON + DAY), ('osi405', -1, NOON)):
        path = _file(tmp_path, product, end=target)
        drift = _product(product, [path])
        x, y = np.meshgrid([-50000., 50000.], [50000., -50000.])
        drift.get_ice_drift(path, {'xc': x, 'yc': y, 'ice_conc': x * 0 + 100})
        frame = processor.apply_drift_correction(1, frame, drift, SIC, None, 0., direction,
                                                 target_time=(target - EPOCH).total_seconds())
    np.testing.assert_allclose(frame[['xu', 'yu']].values, 0, atol=1e-6)


def test_selection_prefers_coverage_before_priority_and_then_priority(tmp_path):
    end = NOON + DAY
    stale = _product('osi455', [_file(tmp_path, 'osi455', end=end - 3 * DAY)])
    exact = _product('osi435', [_file(tmp_path, 'osi435', end=end)])
    wide = _product('osi405', [_file(tmp_path, 'osi405', end=end + DAY)])
    selected = select_drift_product([stale, exact, wide], NOON, end)
    assert selected is exact and not selected.used_temporal_fallback
    assert select_drift_product([wide, exact], NOON, end) is wide


def test_selection_48h_tie_prefers_earlier_interval_across_new_year(tmp_path):
    end = NOON + DAY
    first = _file(tmp_path, 'osi405', end=end)
    second = _file(tmp_path, 'osi405', end=end + DAY)
    drift = _product('osi405', [second, first])
    selected = select_drift_product([drift], NOON, end)
    assert selected.target_files == str(first)
    assert not selected.used_temporal_fallback


def test_selection_fallback_uses_closest_midpoint_and_resets_stale_state(tmp_path):
    end = NOON + DAY
    older = _product('osi455', [_file(tmp_path, 'osi455', end=end - 3 * DAY)])
    closer = _product('osi435', [_file(tmp_path, 'osi435', end=end - DAY)])
    selected = select_drift_product([older, closer], NOON, end)
    assert selected is closer and selected.used_temporal_fallback
    assert selected.target_time_offset_days == -1
    assert select_drift_product([older, closer], NOON + 30 * DAY, end + 30 * DAY) is None
    assert older.target_files is closer.target_files is None


def test_actual_metadata_does_not_promote_partial_coverage_to_exact(tmp_path):
    end = NOON + DAY
    drift = _product('osi455', [_file(tmp_path, 'osi455', end=end, hours=18)])
    selected = select_drift_product([drift], NOON, end)
    assert selected.used_temporal_fallback
    assert selected.target_native_interval == (end - dt.timedelta(hours=18), end)


def test_sync_indexes_both_48h_files_covering_a_daily_step(tmp_path, monkeypatch):
    paths = [_file(tmp_path, 'osi405', end=NOON + n * DAY) for n in (1, 2, 3)]
    spec = input_data._auxiliary_spec(SeaIceDriftProducts, 'osi405', 'nh')
    monkeypatch.setattr(input_data, '_list_ftp_directory', lambda url, timeout: (
        [url + '/' + path.name for path in paths] if url.endswith('/2021/01') else []))
    remote = input_data._inventory_remote('ftp://example.test', [dt.datetime(2021, 1, 1)], spec, 1)
    assert [item.date for item in remote] == [NOON + DAY, NOON + 2 * DAY]
    product = _product('osi405', [path for path in paths if path.name in {r.url.rsplit('/', 1)[1] for r in remote}])
    assert select_drift_product([product], NOON, NOON + DAY).target_files == str(paths[0])


@pytest.mark.parametrize('x_reverse', [False, True])
@pytest.mark.parametrize('y_reverse', [False, True])
def test_velocity_interpolation_preserves_affine_flow_and_coordinate_direction(x_reverse, y_reverse):
    x, y = np.arange(-3, 4) * 25000., np.arange(-3, 4) * 50000.
    x, y = x[::-1] if x_reverse else x, y[::-1] if y_reverse else y
    xx, yy = np.meshgrid(x, y)
    product = _product('osi455')
    product.ice_drift = {'xc': xx, 'yc': yy, 'u': 100 + .001 * xx + .002 * yy,
                         'v': -50 + .003 * xx - .004 * yy, 'velocity_unc': xx * 0 + 10}
    qx, qy = np.array([-12500., 12500.]), np.array([7500., -7500.])
    u, v, sigma = product.drift_correction(qx, qy)
    np.testing.assert_allclose(u, 100 + .001 * qx + .002 * qy)
    np.testing.assert_allclose(v, -50 + .003 * qx - .004 * qy)
    np.testing.assert_allclose(sigma, 10)


@pytest.mark.parametrize('corruption', ['reversed_bounds', 'missing_units'])
def test_invalid_native_timing_is_rejected(tmp_path, corruption):
    path = _file(tmp_path, 'osi405')
    with netCDF4.Dataset(path, 'a') as ds:
        if corruption == 'reversed_bounds':
            ds['time_bnds'][:] = ds['time_bnds'][:][:, ::-1]
        else:
            ds['time'].delncattr('units')
    with pytest.raises(ValueError):
        _product('osi405', [path]).time_interval(str(path))


@pytest.mark.parametrize('direction', ['f', 'r'])
@pytest.mark.parametrize('hour', [6, 12])
def test_stacking_csv_positions_across_product_switch_and_new_year(
    tmp_path, workflow_configs, monkeypatch, direction, hour,
):
    config, = workflow.stacking_configs(workflow_configs['stacking'], hemisphere='nh',
                                        sensors=['cryosat2'], start='2020-12-31', end='2021-01-02')
    files = tmp_path / 'drift'
    _file(files, 'osi455', end=NOON + DAY)
    _file(files, 'osi405', end=NOON + 3 * DAY)
    config['options']['ice_conc_products'] = ['sic']
    config['options']['ice_drift_products'] = ['osi455', 'osi405']
    config['auxiliary']['ice_conc'] = {'sic': str(tmp_path)}
    config['auxiliary']['ice_drift'] = {p: str(files / p) for p in ('osi455', 'osi405')}
    config['output_dir']['trajectories'] = str(tmp_path / 'csv')
    (tmp_path / 'csv').mkdir()
    config['stacking']['thermo_change']['enabled'] = False

    class Altimetry:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def get_file_list(self, directories):
            pass

        def get_file_dates(self):
            pass

        def get_target_files(self, first, last):
            self.date = first
            self.target_files = {'cryosat2': ['altimetry.nc']}

        def get_product(self, sensors):
            self.product = pd.DataFrame({'sensor': ['cryosat2']})

    class Concentration:
        product_id = 'sic'
        file_dates = [NOON + n * DAY for n in range(-1, 5)]
        file_list = [f'sic-{n}.nc' for n in range(-1, 5)]

        @classmethod
        def load_products(cls, *args, **kwargs):
            return [cls()]

        def get_ice_concentration(self, path):
            x, y = np.meshgrid([-75000., -25000., 25000., 75000.],
                               [75000., 25000., -25000., -75000.])
            return {'xc': x, 'yc': y, 'ice_conc': x * 0 + 100}

        def interp_ice_concentration(self, data, x, y):
            return np.ones(len(x))

    def baseline(processor, *args, **kwargs):
        parcel = _parcels(hour)
        clock = (processor.parent.date.replace(hour=hour) - EPOCH).total_seconds()
        parcel['t0'] = parcel['time_current'] = clock
        parcel['sea_ice_thickness'] = 1.
        parcel['source_date'] = processor.parent.date.strftime('%Y-%m-%d')
        processor.master[processor.i][0] = parcel

    monkeypatch.setattr(pipeline, 'init_logger', lambda c: None)
    monkeypatch.setattr(pipeline, 'SeaIceThicknessMultiProducts', Altimetry)
    monkeypatch.setattr(pipeline, 'SeaIceConcentrationProducts', Concentration)
    monkeypatch.setattr(pipeline.DriftAwareProcessor, 'baseline_proc', baseline)
    pipeline.process_direction(config, direction, None)
    target_day = '20210102' if direction == 'f' else '20201231'
    path, = (tmp_path / 'csv').glob(f'*{target_day}*.csv')
    data = read_dasit_csv(path)
    source_day = '2020-12-31' if direction == 'f' else '2021-01-02'
    selected = data.loc[data.source_date == source_day]
    assert len(selected) == 1
    row = selected.iloc[0]
    target = NOON + (2 * DAY if direction == 'f' else dt.timedelta(0))
    source = dt.datetime.fromisoformat(source_day).replace(hour=hour)
    elapsed = (target - source).total_seconds() / 3600
    last = row.geometry.geoms[-1]
    np.testing.assert_allclose([last.x, last.y], [100 * elapsed, -50 * elapsed], atol=1)
    assert {'osi455', 'osi405'} <= set(row['_ice_drift_products'].split('|'))
    assert 'time_current' not in data
