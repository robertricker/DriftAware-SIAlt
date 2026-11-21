import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import MultiPoint, Point, box

from driftaware_sialt.config_loader import load_config
from driftaware_sialt.gridding import gridding
from driftaware_sialt.gridding.deformation import RATE_FIELDS, mean_trajectory_rates
from driftaware_sialt.gridding.gridding_lib import grid_data
from driftaware_sialt.io_tools import write_dasit_csv


REFERENCE = dt.datetime(2020, 1, 2, 12)
EPOCH = dt.datetime(1970, 1, 1)


def _parcel(lag, divergence=.02, shear=.03, hour=12, count=1):
    acquisition = (REFERENCE + dt.timedelta(days=lag)).replace(hour=hour)
    return {
        't0': (acquisition - EPOCH).total_seconds(), 'dt_days': lag,
        'divergence': [0.] + [divergence] * abs(lag),
        'shear': [0.] + [shear] * abs(lag), 'cryosat2_cnt': count,
    }


@pytest.mark.parametrize('lag', [-15, -1, 1, 15])
@pytest.mark.parametrize('hour', [0, 6, 12, 23])
@pytest.mark.parametrize('divergence', [-.02, 0., .02])
def test_constant_strain_mean_has_no_zero_bias_or_backward_sign_flip(lag, hour, divergence):
    data = pd.DataFrame([_parcel(lag, divergence, hour=hour)])
    original = data.copy(deep=True)
    result = mean_trajectory_rates(data, REFERENCE)
    np.testing.assert_allclose(result.iloc[0], [divergence, .03, np.hypot(divergence, .03)])
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize('lag, hour, expected', [(-2, 6, 1/90), (2, 6, -1/70), (-2, 12, 0)])
def test_actual_first_step_weights_rates_and_deformation_before_averaging(lag, hour, expected):
    row = _parcel(lag, shear=0., hour=hour)
    row['divergence'] = [0., .1, -.1]
    result = mean_trajectory_rates(pd.DataFrame([row]), REFERENCE).iloc[0]
    assert result['divergence'] == pytest.approx(expected)
    assert result['shear'] == 0
    assert result['deformation'] == pytest.approx(.1)


def test_missing_or_negative_rates_do_not_invalidate_other_rate_histories():
    data = pd.DataFrame([_parcel(0), _parcel(-1, np.nan), _parcel(-1, shear=-.1)])
    result = mean_trajectory_rates(data, REFERENCE)
    assert result.iloc[0].isna().all()
    assert np.isnan(result.loc[1, 'divergence'])
    assert result.loc[1, 'shear'] == pytest.approx(.03)
    assert result.loc[2, 'divergence'] == pytest.approx(.02)
    assert np.isnan(result.loc[2, 'shear'])
    assert result['deformation'].isna().all()


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
        mean_trajectory_rates(pd.DataFrame([row]), REFERENCE)


@pytest.mark.parametrize('weight', [None, 'dt_days', ['dt_days'], 'counts'])
def test_spatial_rate_means_exclude_zero_lag_and_use_per_variable_denominators(weight):
    rows = [_parcel(-1, .02, .03), _parcel(2, .04, .06, count=9),
            _parcel(0, count=100), _parcel(-1, np.nan, .03)]
    data = gpd.GeoDataFrame(rows, geometry=[Point(2, 2)] * 3 + [Point(12, 2)], crs='EPSG:6931')
    data['thickness'] = [2., 4., 8., 6.]
    data[list(RATE_FIELDS)] = mean_trajectory_rates(data, REFERENCE)
    grid = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=data.crs)
    columns = [*RATE_FIELDS, 'thickness', 'cryosat2_cnt', 'dt_days']
    result = grid_data(data, grid, columns, columns, fill_nan=True,
                       agg_mode=['mean' if weight is None else 'weighted_mean'], weight_var=weight)
    weights = [1., 3., 10.] if weight == 'counts' else ([1/4, 1/9, 1.] if weight is not None else [1., 1., 1.])
    assert result.iloc[0]['divergence'] == pytest.approx(np.average([.02, .04], weights=weights[:2]))
    assert result.iloc[0]['shear'] == pytest.approx(np.average([.03, .06], weights=weights[:2]))
    assert result.iloc[0]['deformation'] == pytest.approx(np.average([np.hypot(.02, .03), np.hypot(.04, .06)], weights=weights[:2]))
    assert result.iloc[0]['thickness'] == pytest.approx(np.average([2., 4., 8.], weights=weights))
    assert np.isnan(result.iloc[1]['divergence'])
    assert result.iloc[1]['shear'] == pytest.approx(.03)
    assert np.isnan(result.iloc[1]['deformation'])
    assert result.iloc[1]['thickness'] == 6.


@pytest.mark.parametrize('hemisphere, mode, weight, zero_only', [
    ('nh', 'da', None, False), ('sh', 'da', 'dt_days', False),
    ('nh', 'cv', 'counts', False), ('sh', 'da', None, True),
])
def test_csv_to_netcdf_corrects_rates_without_extra_fields_or_thickness_changes(
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

    def legacy_means(parcels, reference_time):
        return pd.DataFrame({
            'divergence': parcels['divergence'].apply(np.mean),
            'shear': parcels['shear'].apply(np.mean),
            'deformation': parcels.apply(lambda row: np.mean(np.hypot(row['divergence'], row['shear'])), axis=1),
        })

    outputs = []
    for means in (legacy_means, mean_trajectory_rates):
        folder = tmp_path / means.__name__
        folder.mkdir()
        config['output_dir']['gridded_data'] = str(folder)
        monkeypatch.setattr(gridding, 'mean_trajectory_rates', means)
        gridding.process_file(config, [str(file)], grid, np.array([[1]]), region, source_stack)
        path, = folder.glob('*.nc')
        with xr.open_dataset(path, decode_timedelta=False) as ds:
            outputs.append(ds.load())
    baseline, result = outputs
    changed = {*RATE_FIELDS, 'sea_ice_thickness_dynamic_tendency', 'sea_ice_thickness_thermodynamic_tendency'}
    for name in baseline.variables.keys() - changed:
        xr.testing.assert_identical(result[name], baseline[name])
    assert not any(name.startswith('trajectory_') for name in result.variables)
    for name in RATE_FIELDS:
        assert result[name].dims == ('time', 'yc', 'xc')
        assert result[name].attrs['units'] == 'day-1'
    if zero_only:
        assert all(result[name].isnull().all() for name in changed)
        assert result.sea_ice_thickness.item() == 1.
    else:
        weights = [1., 3., 1.] if weight == 'counts' else ([1/4, 1/9, 1/4] if weight is not None else [1., 1., 1.])
        assert result.divergence.item() == pytest.approx(np.average([.02, .04, .02], weights=weights))
        assert result.shear.item() == pytest.approx(.03)
        assert result.deformation.item() == pytest.approx(np.average([np.hypot(.02, .03), .05, np.hypot(.02, .03)], weights=weights))
        if mode == 'da':
            dynamic = np.average([-1 * .02, -2 * .04, -3 * .02], weights=weights)
            assert result.sea_ice_thickness_dynamic_tendency.item() == pytest.approx(dynamic)
            assert result.sea_ice_thickness_thermodynamic_tendency.item() == pytest.approx(-dynamic)
