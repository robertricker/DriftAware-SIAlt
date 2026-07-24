import geopandas as gpd
import pandas as pd

import shapely as shp
import numpy as np
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore")


def define_grid(bounds, n_cells, epsg, grid_type='rectangular'):
    xmin, xmax = bounds[0], bounds[2]
    ymin, ymax = bounds[1], bounds[3]

    if grid_type == 'rectangular':
        cell_size = (xmax - xmin) / n_cells
        grid_cells = []
        for x0 in np.arange(xmin, xmax, cell_size).tolist():
            for y0 in np.arange(ymin, ymax, cell_size).tolist():
                # bounds
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                grid_cells.append(shp.geometry.box(x0, y0, x1, y1))
        return gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=epsg), cell_size

    elif grid_type == 'circular':
        cell_spacing = (xmax - xmin) / n_cells
        radius = cell_spacing * np.sqrt(2) / 2
        grid_cells = []
        for x_center in np.arange(xmin + cell_spacing/2, xmax, cell_spacing).tolist():
            for y_center in np.arange(ymin + cell_spacing/2, ymax, cell_spacing).tolist():
                # Create a circular cell centered at (x_center, y_center)
                cell = Point(x_center, y_center).buffer(radius)
                grid_cells.append(cell)

        # Create a GeoDataFrame from the grid cells
        return gpd.GeoDataFrame(geometry=grid_cells, crs=epsg), radius * 2

    else:
        raise ValueError("Invalid grid_type. Use 'rectangular' or 'circular'.")


def grid_data(gdf, grid, var, var_str, hist_n_bins=None, hist_range=None, fill_nan=False, agg_mode=None, weight_var=None):
    if agg_mode is None:
        agg_mode = ['mean', 'std']
    tmp_grid = grid.copy()
    #tmp_grid_unc = grid.copy()
    merged = gpd.sjoin(gdf[var + ['geometry']].copy(), grid, how='left', predicate='within')
    if 'weighted_mean' in agg_mode:
        merged['weight'] = 1
        if not weight_var or weight_var=='':
             weight_var = ['dt_days']
        # The following condition is only configured for the case of Sentinel3a, Sentinel3b and Cryosat2 are used using as 'sentinel3a_sentinel3b_cryosat2'
        if (weight_var == 'counts') & ('sentinel3a_cnt' in merged.columns) & ('sentinel3b_cnt' in merged.columns) & ('cryosat2_cnt' in merged.columns):
            if (merged.cryosat2_cnt.sum() != 0) & (merged.sentinel3b_cnt.sum() != 0) & (merged.sentinel3a_cnt.sum() !=0):
                # only where there is sentinel3a and sentinel3b in the same parcel otherwise is should be 0.5 for everyone
                # count for each grid cell the total number of observations per mission
                counts = merged.dissolve(by='index_right',aggfunc={'sentinel3a_cnt': 'sum','sentinel3b_cnt': 'sum','cryosat2_cnt': 'sum'})
                # Capture where there is the three missions in the same grid cell
                three_missions = ((counts['sentinel3a_cnt'] > 0) &
                                  (counts['sentinel3b_cnt'] > 0) &
                                  (counts['cryosat2_cnt'] > 0))
                counts['q_cs2'] = 0.5
                counts['q_s3a'] = 0.5
                counts['q_s3b'] = 0.5
                counts.loc[three_missions, 'q_s3a'] = 0.25
                counts.loc[three_missions, 'q_s3b'] = 0.25
                merged = merged.merge(counts[['q_s3a', 'q_s3b', 'q_cs2']],
                                      left_on='index_right',
                                      right_index=True, how='left')

                merged['weight'] = merged['q_cs2']*np.sqrt(merged['cryosat2_cnt']) \
                                + merged['q_s3a']*np.sqrt(merged['sentinel3b_cnt']) \
                                + merged['q_s3b']*np.sqrt(merged['sentinel3a_cnt'])
                

            else:
                merged['weight'] = 1
            
        else :
            for wv in weight_var:
                stdz_var = (merged[f'{wv}'] - merged[f'{wv}'].mean())/(merged[f'{wv}'].std())
                # add one to the stdzed array to avoid inf values
                merged['weight']*=  1/(stdz_var+1)**2 
        
        # var * weight
        merged_drop = merged.drop(merged[['geometry', 'index_right']], axis=1)
        weighted = gpd.GeoDataFrame(pd.concat([merged_drop*merged['weight'].values[:, None], merged[['geometry', 'index_right']]], axis=1), crs=merged.crs, geometry=merged.geometry)
        # var**2 * weight**2
        #weighted_square = gpd.GeoDataFrame(pd.concat([(merged_drop**2)*(merged['weight'].values[:, None]**2), merged[['geometry', 'index_right']]], axis=1), crs=merged.crs, geometry=merged.geometry)
        # sum weighted values
        dissolve_sum = weighted.dissolve(by='index_right', aggfunc=np.sum)
        # sum squared weighted squared values
        #dissolve_square_sum = weighted_square.dissolve(by='index_right', aggfunc=np.sum)
        # sum the weights
        weight_sum = merged.dissolve(by='index_right', aggfunc=np.sum)['weight'].values[:, None]
        # sum weighted values / sum of weights
        dissolve_weighted_mean = pd.concat([dissolve_sum.drop(dissolve_sum[['geometry']], axis=1)/weight_sum, dissolve_sum[['geometry']]], axis=1)
        # sum squared weighted squared values / sum of weights**2 -> made for uncertainties
        #dissolve_weighted_unc_mean = pd.concat([np.sqrt(dissolve_square_sum.drop(dissolve_square_sum[['geometry']], axis=1)/(weight_sum**2)), dissolve_sum[['geometry']]], axis=1)
           

        for i in range(0, len(var)):
            if np.size(dissolve_weighted_mean[var[i]].shape)>1:
                diss = dissolve_weighted_mean[var[i]].values[:, 0]
            else: 
                diss = dissolve_weighted_mean[var[i]].values
            tmp_grid.loc[dissolve_weighted_mean.index, var_str[i]] = diss
            #tmp_grid_unc.loc[dissolve_weighted_unc_mean.index, var_str[i]] = dissolve_weighted_unc_mean[var[i]].values
        if not fill_nan:
            tmp_grid = tmp_grid.dropna()
            #tmp_grid_unc = tmp_grid_unc.dropna()
        centroidseries = tmp_grid['geometry'].centroid
        tmp_grid['x'], tmp_grid['y'] = centroidseries.x, centroidseries.y
        tmp_grid = tmp_grid.set_index(['x', 'y'])
        #tmp_grid_unc['x'], tmp_grid_unc['y'] = centroidseries.x, centroidseries.y
        #tmp_grid_unc = tmp_grid_unc.set_index(['x', 'y'])
        #return tmp_grid, tmp_grid_unc
    
    if 'mean' in agg_mode:
        dissolve_mean = merged.dissolve(by='index_right', aggfunc=np.mean)
        for i in range(0, len(var)):
            tmp_grid.loc[dissolve_mean.index, var_str[i]] = dissolve_mean[var[i]].values
    if 'std' in agg_mode:
        dissolve_std = merged.dissolve(by='index_right', aggfunc=np.std)
        for i in range(0, len(var)):
            tmp_grid.loc[dissolve_std.index, var_str[i] + '_std'] = dissolve_std[var[i]].values
    if 'sum' in agg_mode:
        dissolve_sum = merged.dissolve(by='index_right', aggfunc=np.sum)
        for i in range(0, len(var)):
            tmp_grid.loc[dissolve_sum.index, var_str[i] + '_sum'] = dissolve_sum[var[i]].values
    if 'cnt' in agg_mode:
        dissolve_cnt = merged.dissolve(by='index_right', aggfunc='count')
        for i in range(0, len(var)):
            tmp_grid.loc[dissolve_cnt.index, var_str[i] + '_cnt'] = dissolve_cnt[var[i]].values
    if 'mode' in agg_mode:
        dissolve_mode = merged.dissolve(by='index_right',
                                        aggfunc=lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        for i in range(len(var)):
            tmp_grid.loc[dissolve_mode.index, var_str[i]] = dissolve_mode[var[i]].values
    if 'hist' in agg_mode:
        dissolve_hist = merged.dissolve(by='index_right', aggfunc=np.mean)
        dissolve_hist.reset_index(inplace=True)
        dissolve_hist[var[0]] = ''
        for i in range(0, len(var)):
            for j in dissolve_hist.index_right.unique():
                tmp = merged[merged.index_right == j].drop(['index_right', 'geometry'], axis=1).reset_index(drop=True)
                tmp_hist = np.histogram(np.array(tmp[var[i]]),
                                        bins=hist_n_bins,
                                        range=(hist_range[0], hist_range[1]))[0]
                dissolve_hist[var[i]][dissolve_hist.index[dissolve_hist.index_right == j][0]] = ' '.join(
                    map(str, tmp_hist))

            tmp_grid.loc[dissolve_hist.index_right, var_str[i] + '_hist'] = dissolve_hist[var[i]].values

    if not fill_nan:
        tmp_grid = tmp_grid.dropna()

    centroidseries = tmp_grid['geometry'].centroid
    tmp_grid['x'], tmp_grid['y'] = centroidseries.x, centroidseries.y
    tmp_grid = tmp_grid.set_index(['x', 'y'])
    return tmp_grid


def modal_var(df, n_bins, bin_size, hist_range):
    bin_center = np.arange(hist_range[0] + bin_size / 2, hist_range[1] + bin_size / 2, bin_size)
    id_mode = df.loc[:, '0':str(n_bins-1)].idxmax(axis=1)
    return bin_center[np.array(id_mode.astype(int))]
