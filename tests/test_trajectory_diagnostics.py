import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import MultiPoint, Point, box

from driftaware_sialt.config_loader import load_config
from driftaware_sialt.gridding import gridding
from driftaware_sialt.gridding.trajectory_diagnostics import (
    COUNT_FIELD, INTERVALS, MEAN_FIELDS, add_trajectory_diagnostics,
    integrate_trajectories,
)
from driftaware_sialt.io_tools import write_dasit_csv


REFERENCE = dt.datetime(2020, 1, 2, 12)
EPOCH = dt.datetime(1970, 1, 1)


def _parcel(lag, divergence=.02, shear=.03, hour=12, count=1):
    direction = -np.sign(lag)
    acquisition = REFERENCE - dt.timedelta(days=int(direction * abs(lag)))
    acquisition = acquisition.replace(hour=hour)
    return {
        't0': (acquisition - EPOCH).total_seconds(), 'dt_days': lag,
        'divergence': [0.] + [divergence] * abs(lag),
        'shear': [0.] + [shear] * abs(lag), 'cryosat2_cnt': count,
    }


@pytest.mark.parametrize('direction', [1, -1])
@pytest.mark.parametrize('hour', [0, 6, 12, 23])
@pytest.mark.parametrize('lag', [1, 15])
def test_constant_strain_integrals_use_actual_duration_and_signed_time(direction, hour, lag):
    data = pd.DataFrame([_parcel(-direction * lag, hour=hour)])
    original = data.copy(deep=True)
    result = integrate_trajectories(data, REFERENCE).iloc[0]
    duration = lag + direction * (12 - hour) / 24
    np.testing.assert_allclose(result[list(MEAN_FIELDS)].to_numpy(dtype=float), [
        direction * .02 * duration, .02 * duration, 0., .03 * duration,
        np.hypot(.02, .03) * duration, duration, np.hypot(.02, .03),
    ])
    assert result['trajectory_interval'] == INTERVALS[direction < 0]
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize('lag', [-2, 2])
def test_opening_and_closing_do_not_cancel_in_total_exposure(lag):
    row = _parcel(lag, shear=0.)
    row['divergence'] = [0., .1, -.1]
    result = integrate_trajectories(pd.DataFrame([row]), REFERENCE).iloc[0]
    assert result['trajectory_integrated_divergence'] == pytest.approx(0)
    assert result['trajectory_accumulated_opening'] == pytest.approx(.1)
    assert result['trajectory_accumulated_convergence'] == pytest.approx(.1)
    assert result['trajectory_accumulated_deformation'] == pytest.approx(.2)


def test_partial_first_step_weights_the_rate_at_that_step_only():
    row = _parcel(-2, shear=0., hour=6)
    row['divergence'] = [0., .1, -.1]
    result = integrate_trajectories(pd.DataFrame([row]), REFERENCE).iloc[0]
    assert result['trajectory_integrated_divergence'] == pytest.approx(.025)
    assert result['trajectory_accumulated_opening'] == pytest.approx(.125)
    assert result['trajectory_accumulated_convergence'] == pytest.approx(.1)
    assert result['trajectory_accumulated_deformation'] == pytest.approx(.225)


@pytest.mark.parametrize('change', ['length', 'initial_zero', 'time', 'lag'])
def test_inconsistent_histories_fail_instead_of_inventing_durations(change):
    row = _parcel(-2)
    if change == 'length':
        row['shear'] = [0., .03]
    elif change == 'initial_zero':
        row['divergence'][0] = .1
    elif change == 'time':
        row['t0'] += 5 * 86400
    else:
        row['dt_days'] = -1.5
    with pytest.raises(ValueError, match='trajectory'):
        integrate_trajectories(pd.DataFrame([row]), REFERENCE)


@pytest.mark.parametrize('weight', [None, 'dt_days', ['dt_days'], 'counts'])
def test_gridding_separates_intervals_and_excludes_unsupported_histories(weight):
    rows = [_parcel(-1, shear=0., hour=0), _parcel(-2, .04, 0., count=9),
            _parcel(1, shear=0.), _parcel(0), _parcel(-1, np.nan), _parcel(-1, shear=-.1)]
    geometry = [Point(2, 2)] * len(rows) + [Point(100, 100)]
    rows.append(_parcel(-1, divergence=.9))
    data = gpd.GeoDataFrame(rows, geometry=geometry, crs='EPSG:6931')
    grid = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=data.crs)
    dataset = xr.Dataset({'sea_ice_thickness': (('time', 'yc', 'xc'), [[[2., 4.]]])},
                         coords={'time': [0.], 'yc': [5.], 'xc': [5., 15.]})
    result = add_trajectory_diagnostics(
        dataset, data, integrate_trajectories(data, REFERENCE), grid,
        is_weight=weight is not None, weight_var=weight)
    assert result.trajectory_interval.values.tolist() == list(INTERVALS)
    assert result[COUNT_FIELD].dtype == np.int32
    before = result.sel(trajectory_interval='before_reference', xc=5.)
    after = result.sel(trajectory_interval='after_reference', xc=5.)
    weights = [1., 3.] if weight == 'counts' else ([1/4, 1/9] if weight is not None else [1., 1.])
    assert before.trajectory_integrated_divergence.item() == pytest.approx(np.average([.03, .08], weights=weights))
    assert before.trajectory_mean_deformation_rate.item() == pytest.approx(np.average([.02, .04], weights=weights))
    assert before[COUNT_FIELD].item() == 2
    assert after[COUNT_FIELD].item() == 1
    assert after.trajectory_integrated_divergence.item() == pytest.approx(-.02)
    assert after.trajectory_accumulated_opening.item() == pytest.approx(.02)
    assert (result[COUNT_FIELD].sel(xc=15.) == 0).all()
    assert all(result[name].sel(xc=15.).isnull().all() for name in MEAN_FIELDS)
    xr.testing.assert_identical(result.sea_ice_thickness, dataset.sea_ice_thickness)


def test_zero_lag_only_has_no_deformation_history_in_either_interval():
    data = gpd.GeoDataFrame([_parcel(0)], geometry=[Point(2, 2)], crs='EPSG:6931')
    grid = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=data.crs)
    dataset = xr.Dataset(coords={'time': [0.], 'yc': [5.], 'xc': [5.]})
    diagnostics = integrate_trajectories(data, REFERENCE)
    assert diagnostics.trajectory_accumulated_deformation.iloc[0] == 0
    result = add_trajectory_diagnostics(dataset, data, diagnostics, grid)
    assert (result[COUNT_FIELD] == 0).all()
    assert all(result[name].isnull().all() for name in MEAN_FIELDS)


@pytest.mark.parametrize('hemisphere, mode, weight, zero_only', [
    ('nh', 'da', None, False), ('sh', 'da', 'dt_days', False),
    ('nh', 'cv', 'counts', False), ('sh', 'da', None, True),
])
def test_csv_to_netcdf_adds_diagnostics_without_changing_existing_fields(
        workflow_configs, tmp_path, monkeypatch, hemisphere, mode, weight, zero_only):
    config = load_config(workflow_configs['gridding'], hemisphere=hemisphere, sensors=['cryosat2'])
    config['gridding']['mode'] = mode
    config['gridding']['weighting'].update(is_weight=weight is not None, var_to_weight_with=weight)
    stack = {'mode': 'fr', 't_window': 16,
             'hist': {'n_bins': 2, 'range': {'thickness': [0, 10], 'freeboard': [0, 1]}}}
    source_stack = {'mode': 'fr', 'window_days': 16, 'histogram': stack['hist']}
    rows = [_parcel(-1, hour=6), _parcel(-2, .04, count=9), _parcel(1), _parcel(0)]
    if zero_only:
        rows = rows[-1:]
    for i, row in enumerate(rows):
        row.update(sea_ice_thickness=float(i + 1), sea_ice_thickness_uncorrected=float(i + 1),
                   sea_ice_thickness_hist='1 0', sea_ice_thickness_growth_unc=.1,
                   sea_ice_thickness_drift_unc=.1, sea_ice_thickness_parcel_unc=.1,
                   growth=0., growth_interpolated=0., ice_conc=.9, cryosat2=1.)
        row['divergence'] = ' '.join(map(str, row['divergence']))
        row['shear'] = ' '.join(map(str, row['shear']))
    data = gpd.GeoDataFrame(rows, geometry=[MultiPoint([(0, 0)] * (abs(row['dt_days']) + 1))
                                          for row in rows], crs=config['options']['out_epsg'])
    file = tmp_path / f'ESACCI-SEAICE-L2P-DA-SITHICK-SIRAL_CRYOSAT2-{hemisphere.upper()}-20200102-fv2.0.csv'
    write_dasit_csv(data, file, {'options': config['options'], 'stacking': stack})
    grid = gpd.GeoDataFrame(geometry=[box(-100, -100, 100, 100)], crs=data.crs)
    region = {'flag_values': np.array([1], dtype=np.int32),
              'flag_meanings': 'test_region', 'source': 'fixture', 'long_name': 'test region'}
    monkeypatch.setattr(gridding, 'init_logger', lambda config: None)
    monkeypatch.setattr(gridding, 'add_full_domain_ice_concentration', lambda ds, *args: ds)
    original = gridding.add_trajectory_diagnostics
    outputs = []
    for enabled in (False, True):
        folder = tmp_path / str(enabled)
        folder.mkdir()
        config['output_dir']['gridded_data'] = str(folder)
        monkeypatch.setattr(gridding, 'add_trajectory_diagnostics', original if enabled else lambda ds, *a, **k: ds)
        gridding.process_file(config, [str(file)], grid, np.array([[1]]), region, source_stack)
        path, = folder.glob('*.nc')
        with xr.open_dataset(path, decode_timedelta=False) as ds:
            outputs.append(ds.load())
    baseline, result = outputs
    for name in baseline.variables:
        xr.testing.assert_identical(result[name], baseline[name])
    assert result[COUNT_FIELD].sel(trajectory_interval='before_reference').item() == (0 if zero_only else 2)
    assert result[COUNT_FIELD].sel(trajectory_interval='after_reference').item() == (0 if zero_only else 1)
    for name in (*MEAN_FIELDS, COUNT_FIELD):
        assert result[name].dims == ('time', 'trajectory_interval', 'yc', 'xc')
        assert result[name].attrs['long_name']
        assert result[name].attrs['units']
    assert result.trajectory_accumulated_opening.attrs['units'] == '1'
    assert result.trajectory_duration.attrs['units'] == 'days'
    assert result.trajectory_mean_deformation_rate.attrs['units'] == 'day-1'
    if zero_only:
        assert all(result[name].isnull().all() for name in MEAN_FIELDS)
    else:
        assert result.trajectory_integrated_divergence.sel(trajectory_interval='after_reference').item() < 0
