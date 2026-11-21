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
    merged = gpd.sjoin(gdf[var + ['geometry']].copy(), grid, how='left', predicate='within')
    if 'weighted_mean' in agg_mode:
        merged['weight'] = 1
        if not weight_var or weight_var=='':
             weight_var = 'dt_days'
        if weight_var == 'counts':
            count_columns = [
                column for column in merged.columns
                if column.endswith('_cnt')]
            if not count_columns:
                raise ValueError(
                    "count weighting requires sensor columns ending in '_cnt'")

            counts = merged.dissolve(
                by='index_right',
                aggfunc={column: 'sum' for column in count_columns})
            sentinel_columns = [
                column for column in ('sentinel3a_cnt', 'sentinel3b_cnt')
                if column in count_columns]
            mission_groups = [
                [column] for column in count_columns
                if column not in sentinel_columns]
            if sentinel_columns:
                mission_groups.append(sentinel_columns)

            group_states = []
            number_of_groups = pd.Series(0, index=counts.index, dtype=int)
            for group in mission_groups:
                active_missions = counts[group].gt(0)
                active_count = active_missions.sum(axis=1)
                group_active = active_count.gt(0)
                number_of_groups += group_active.astype(int)
                group_states.append(
                    (group, active_missions, active_count, group_active))

            share_columns = []
            for group, active_missions, active_count, group_active in group_states:
                group_share = pd.Series(0.0, index=counts.index)
                group_share.loc[group_active] = (
                    1.0 / number_of_groups.loc[group_active])
                for column in group:
                    share_column = '_share_' + column
                    share_columns.append(share_column)
                    counts[share_column] = 0.0
                    mission_active = active_missions[column]
                    counts.loc[mission_active, share_column] = (
                        group_share.loc[mission_active]
                        / active_count.loc[mission_active])

            merged = merged.merge(
                counts[share_columns],
                left_on='index_right', right_index=True, how='left')
            merged['weight'] = sum(
                merged['_share_' + column]
                * np.sqrt(merged[column].clip(lower=0))
                for column in count_columns)
            
        else :
            weight_vars = (
                [weight_var] if isinstance(weight_var, str) else weight_var)
            for wv in weight_vars:
                if wv == 'dt_days':
                    merged['weight'] *= 1 / (1 + merged[wv].abs())**2
                else:
                    stdz_var = (
                        (merged[wv] - merged[wv].mean())
                        / merged[wv].std())
                    merged['weight'] *= 1 / (stdz_var + 1)**2
        
        merged_drop = merged.drop(merged[['geometry', 'index_right']], axis=1)
        weighted = gpd.GeoDataFrame(pd.concat([merged_drop*merged['weight'].values[:, None], merged[['geometry', 'index_right']]], axis=1), crs=merged.crs, geometry=merged.geometry)
        # sum weighted values
        dissolve_sum = weighted.dissolve(by='index_right', aggfunc=np.sum)
        # Each variable needs its own denominator: missing rates must not dilute
        # the mean or remove otherwise valid thickness observations.
        weight_sum = merged[var].notna().mul(merged['weight'], axis=0)
        weight_sum = weight_sum.groupby(merged['index_right']).sum().replace(0, np.nan)
        dissolve_weighted_mean = pd.concat([
            dissolve_sum[var] / weight_sum, dissolve_sum[['geometry']]], axis=1)
           

        for i in range(0, len(var)):
            if np.size(dissolve_weighted_mean[var[i]].shape)>1:
                diss = dissolve_weighted_mean[var[i]].values[:, 0]
            else: 
                diss = dissolve_weighted_mean[var[i]].values
            tmp_grid.loc[dissolve_weighted_mean.index, var_str[i]] = diss
        if not fill_nan:
            tmp_grid = tmp_grid.dropna()
        centroidseries = tmp_grid['geometry'].centroid
        tmp_grid['x'], tmp_grid['y'] = centroidseries.x, centroidseries.y
        tmp_grid = tmp_grid.set_index(['x', 'y'])
    
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
