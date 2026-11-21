"""Diagnostic deformation exposure between acquisition and reference time."""

import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from driftaware_sialt.gridding.gridding_lib import grid_data


INTERVALS = ('before_reference', 'after_reference')
MEAN_FIELDS = (
    'trajectory_integrated_divergence',
    'trajectory_accumulated_opening',
    'trajectory_accumulated_convergence',
    'trajectory_accumulated_shear',
    'trajectory_accumulated_deformation',
    'trajectory_duration',
    'trajectory_mean_deformation_rate',
)
COUNT_FIELD = 'trajectory_sample_count'


def integrate_trajectories(data, reference_time):
    """Integrate stored day-1 rates without changing the input parcels.

    Histories contain one initialization zero followed by one sample per step.
    All steps end at noon. After the first, steps last exactly one day, so t0,
    the reference noon and the signed acquisition-day lag recover their lengths.
    Zero-lag parcels have no sampled interval and belong to neither group.
    """
    result = pd.DataFrame(np.nan, index=data.index, columns=MEAN_FIELDS)
    result['trajectory_interval'] = ''
    reference = (reference_time - dt.datetime(1970, 1, 1)).total_seconds()
    lags = data['dt_days'].to_numpy(dtype=float)
    if not np.isfinite(lags).all() or np.any(lags != np.trunc(lags)):
        raise ValueError('trajectory diagnostics require integer acquisition-day lags')

    for lag, group in data.groupby(data['dt_days'].abs()):
        steps = int(lag)
        histories = []
        for name in ('divergence', 'shear'):
            try:
                values = np.asarray(group[name].tolist(), dtype=float)
            except ValueError as error:
                raise ValueError(f'inconsistent trajectory {name} histories') from error
            if values.shape != (len(group), steps + 1) or np.any(values[:, 0] != 0):
                raise ValueError(f'trajectory {name} history must contain an initial zero and {steps} steps')
            histories.append(values[:, 1:])
        if steps == 0:
            result.loc[group.index, MEAN_FIELDS[:-1]] = 0.
            continue

        direction = -np.sign(group['dt_days'].to_numpy(dtype=float))
        duration = direction * (reference - group['t0'].to_numpy(dtype=float)) / 86400.
        first_step = duration - (steps - 1)
        if not np.isfinite(duration).all() or np.any(first_step <= 0) or np.any(first_step > 2):
            raise ValueError('trajectory acquisition time is inconsistent with its lag and reference noon')
        durations = np.ones_like(histories[0])
        durations[:, 0] = first_step
        divergence, shear = histories
        valid = np.isfinite(divergence).all(axis=1) & np.isfinite(shear).all(axis=1)
        valid &= (shear >= 0).all(axis=1)
        def integrate(rate):
            return np.sum(rate * durations, axis=1)

        deformation = integrate(np.hypot(divergence, shear))
        values = np.column_stack((
            direction * integrate(divergence),
            integrate(np.maximum(divergence, 0)),
            integrate(np.maximum(-divergence, 0)),
            integrate(shear),
            deformation,
            duration,
            deformation / duration,
        ))
        result.loc[group.index[valid], MEAN_FIELDS] = values[valid]
        result.loc[group.index[valid], 'trajectory_interval'] = np.where(
            direction[valid] > 0, INTERVALS[0], INTERVALS[1])
    return result


def add_trajectory_diagnostics(dataset, data, diagnostics, grid, *, is_weight=False, weight_var=None):
    """Grid parcel integrals independently within each supported time interval."""
    support = []
    if is_weight:
        if weight_var == 'counts':
            support = [name for name in data.columns if name.endswith('_cnt')]
        else:
            weight_var = weight_var or 'dt_days'
            support = [weight_var] if isinstance(weight_var, str) else weight_var
    columns = list(dict.fromkeys([*MEAN_FIELDS, *support]))
    parcels = gpd.GeoDataFrame(
        pd.concat([data[['geometry', *support]], diagnostics], axis=1),
        geometry='geometry', crs=data.crs)
    parts = []
    for interval in INTERVALS:
        subset = parcels.loc[parcels['trajectory_interval'] == interval]
        if subset.empty:
            result = grid.copy()
            result[list(MEAN_FIELDS)] = np.nan
            result[COUNT_FIELD] = np.int32(0)
        else:
            result = grid_data(
                subset, grid, columns, columns, fill_nan=True,
                agg_mode=['weighted_mean' if is_weight else 'mean'], weight_var=weight_var)
            joined = gpd.sjoin(subset[['geometry']], grid, how='inner', predicate='within')
            counts = joined.groupby('index_right').size().reindex(grid.index, fill_value=0)
            result[COUNT_FIELD] = counts.to_numpy(dtype=np.int32)
        result['xc'] = result.geometry.centroid.x.round()
        result['yc'] = result.geometry.centroid.y.round()
        frame = pd.DataFrame(result).set_index(['yc', 'xc'])[[*MEAN_FIELDS, COUNT_FIELD]]
        part = xr.Dataset.from_dataframe(frame).reindex(yc=dataset.yc.values, xc=dataset.xc.values)
        parts.append(part.expand_dims(time=dataset.time.values))
    gridded = xr.concat(parts, dim=xr.IndexVariable('trajectory_interval', list(INTERVALS)))
    gridded = gridded.transpose('time', 'trajectory_interval', 'yc', 'xc')
    return xr.merge([dataset, gridded])
