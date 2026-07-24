from driftaware_sialt.gridding import gridding_lib
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.stats import linregress
import os
from driftaware_sialt.stacking.point_density_correction import latitude_band_density


def get_land_area_correction(config):
    """Return resolved land-correction settings and fail early if misconfigured."""
    growth_options = config['stacking']['growth_estimation']
    settings = dict(growth_options.get('land_area_correction', {}))
    settings.setdefault('enabled', False)
    settings['path'] = config.get('auxiliary', {}).get('land_mask')
    if settings['enabled'] and (
            not settings['path'] or not os.path.isfile(settings['path'])):
        raise FileNotFoundError(
            'Land-area correction is enabled, but auxiliary.land_mask does not '
            f"point to a file: {settings['path']}")
    return settings


def interpolate_growth(data, interp_var, growth_range, grid, cell_width, min_n_tiepoints, nbs, hem,
                       land_area_correction=None):
    merged = gpd.sjoin(data, grid, how='left', predicate='within')
    tmp = (merged.groupby(['index_right', 'dt_days'], as_index=False)
           .agg({'geometry': 'first', interp_var: 'mean'})
           .pipe(gpd.GeoDataFrame, geometry='geometry', crs=merged.crs))
    n_tiepoints = tmp.groupby('index_right')['dt_days'].count()
    valid_indices = n_tiepoints[n_tiepoints >= min_n_tiepoints].index
    if len(valid_indices) == 0:
        raise ValueError('Number of tie points insufficient for all grid cells')
    tmp = tmp[tmp['index_right'].isin(valid_indices)]
    tmp.set_index('index_right', inplace=True)
    eps = 1.8
    land_options = land_area_correction or {}
    counts = latitude_band_density(
        data,
        hem,
        land_mask_path=land_options.get('path'),
        exclude_land=land_options.get('enabled', False),
    )
    valid_density = counts.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['density_km2_no_land'])
    if valid_density.empty:
        raise ValueError('No latitude band has a positive ocean area for density estimation')
    if len(valid_density) == 1 or valid_density['lat_band'].nunique() == 1:
        slope, intercept = 0.0, valid_density['density_km2_no_land'].iloc[0]
    else:
        slope, intercept, *_ = linregress(
            valid_density['lat_band'].values,
            valid_density['density_km2_no_land'].values)

    def density_to_smoothing(density):
        density = np.clip(np.asarray(density), 0.0, 0.035)
        return np.interp(density, [0.0, 0.035], [80.0, 10.0])
    
    # perform linear fit
    tmp['coeff'] = tmp.groupby('index_right').apply(
        lambda x: np.polyfit(x['dt_days'], x[interp_var], deg=1, cov=True))
    tmp['growth'] = [x[0][0] for x in tmp['coeff'].values]
    tmp['growth_unc'] = [np.sqrt(np.diag(x[1])[0]) for x in tmp['coeff'].values]
    tmp[(tmp['growth'] > growth_range[1]) | (tmp['growth'] < growth_range[0])] = np.nan
    growth_raw = pd.merge(
        merged, tmp[~tmp.index.duplicated(keep='first')].reset_index(),
        left_on='index_right', right_on='index_right', how='left')['growth']
    tmp = tmp.drop(columns=['coeff'])
    tmp.index.names = ['index']

    # gridding of growth coefficients
    growth_grid = gridding_lib.grid_data(tmp, grid, ['growth', 'growth_unc'], ['growth', 'growth_unc'], fill_nan=True)
    centroidseries = growth_grid['geometry'].centroid
    growth_grid['yc'], growth_grid['xc'] = centroidseries.x, centroidseries.y
    arr_density = slope*(np.array(growth_grid.dropna().geometry.centroid.to_crs(4326).geometry.y))+intercept
    arr_positif = np.clip(arr_density, 0.0, 0.035)
    # interpolation of growth for all valid target variable data points
    fg = RBFInterpolator(np.vstack((np.array(growth_grid.dropna()['yc']),
                                    np.array(growth_grid.dropna()['xc']))).transpose(),
                         np.array(growth_grid.dropna()['growth']),
                         neighbors=nbs,
                         smoothing=density_to_smoothing(arr_positif),
                         kernel='gaussian', epsilon=eps/cell_width)
    
    fg_unc = RBFInterpolator(np.vstack((np.array(growth_grid.dropna()['yc']),
                                        np.array(growth_grid.dropna()['xc']))).transpose(),
                             np.array(growth_grid.dropna()['growth_unc']),
                             neighbors=nbs,
                             smoothing=density_to_smoothing(arr_positif),
                             kernel='gaussian', epsilon=eps/cell_width)
    return fg, fg_unc, growth_raw.values, n_tiepoints, counts
