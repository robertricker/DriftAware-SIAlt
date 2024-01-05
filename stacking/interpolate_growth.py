from gridding import gridding_lib
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy import interpolate
from scipy.interpolate import RBFInterpolator


def interpolate_growth(data, interp_var, growth_range, grid, cell_width, min_n_tiepoints, nbs):
    merged = gpd.sjoin(data, grid, how='left', predicate='within')
    tmp = (merged.groupby(['index_right', 'dt_days'], as_index=False)
           .agg({'geometry': 'first', interp_var: 'mean'})
           .pipe(gpd.GeoDataFrame, geometry='geometry', crs=merged.crs))
    n_tiepoints = tmp.groupby('index_right')['dt_days'].count()
    valid_indices = n_tiepoints[n_tiepoints >= min_n_tiepoints].index
    tmp = tmp[tmp['index_right'].isin(valid_indices)]
    tmp.set_index('index_right', inplace=True)
    eps = 1.8
    fsm = interpolate.interp1d(np.array([40.0, 90.0]), np.array([80, 10]))

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

    # interpolation of growth for all valid target variable data points
    fg = RBFInterpolator(np.vstack((np.array(growth_grid.dropna()['yc']),
                                    np.array(growth_grid.dropna()['xc']))).transpose(),
                         np.array(growth_grid.dropna()['growth']),
                         neighbors=nbs,
                         smoothing=fsm(
                             np.array(growth_grid.dropna().geometry.centroid.to_crs(4326).geometry.y)),
                         kernel='gaussian', epsilon=eps/cell_width)

    fg_unc = RBFInterpolator(np.vstack((np.array(growth_grid.dropna()['yc']),
                                        np.array(growth_grid.dropna()['xc']))).transpose(),
                             np.array(growth_grid.dropna()['growth_unc']),
                             neighbors=nbs,
                             smoothing=fsm(
                                 np.array(growth_grid.dropna().geometry.centroid.to_crs(4326).geometry.y)),
                             kernel='gaussian', epsilon=eps/cell_width)
    return fg, fg_unc, growth_raw.values
