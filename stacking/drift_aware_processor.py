from gridding import gridding_lib
import pandas as pd
import geopandas as gpd
import numpy as np
import sys
from loguru import logger
import datetime


class DriftAwareProcessor:
    def __init__(self, parent, **kwargs):

        self.parent = parent
        self.sensor = parent.sensor
        self.target_var = parent.target_var
        self.add_variable = parent.add_variable
        self.out_epsg = parent.out_epsg

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.i = None

    def baseline_proc(self, sic_product, hist_n_bins, hist_range):
        # adds the original measurements at t=0 (without drift correction) to the master structure
        sit = self.parent.product
        sit[self.target_var + '_l2_unc'] **= 2
        if self.sensor == 'icesat2':
            beams = np.array(['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r'])
            for beam in sit.beam.unique():
                tmp = (sit[[self.target_var, self.target_var + '_l2_unc', 'geometry', 'time', 'beam'] + self.add_variable]
                       .copy()
                       .loc[sit['beam'] == beam]
                       .drop(columns=['beam'])
                       )
                tmp = tmp.reset_index(drop=True)
                tmp_grid = gridding_lib.grid_data(tmp, self.grid, [self.target_var], [self.target_var],
                                                  hist_n_bins=hist_n_bins, hist_range=hist_range,
                                                  agg_mode=['mean', 'std', 'hist'])
                unc_grid = gridding_lib.grid_data(tmp, self.grid, [self.target_var+'_l2_unc'],
                                                  [self.target_var + '_l2_unc'], agg_mode=['sum', 'cnt'])
                add_grid = gridding_lib.grid_data(tmp, self.grid, self.add_variable+['time'],
                                                  self.add_variable+['time'], agg_mode=['mean'])

                tmp_grid[self.target_var + '_l2_unc'] = np.sqrt(unc_grid[self.target_var + '_l2_unc_sum']) / unc_grid[
                    self.target_var + '_l2_unc_cnt']
                tmp_grid[self.add_variable] = add_grid[self.add_variable]
                tmp_grid['t0'] = add_grid['time']
                tmp_grid['xu'] = tmp_grid.index.get_level_values('x')
                tmp_grid['yu'] = tmp_grid.index.get_level_values('y')
                tmp_grid['dt_days'] = 0
                tmp_grid['beam'] = beam
                tmp_grid['beam_type'] = sit[sit['beam'] == beam]['beam_type'].iloc[0]
                tmp_grid.reset_index(drop=True, inplace=True)
                tmp_grid = gpd.GeoDataFrame(
                    tmp_grid, geometry=gpd.points_from_xy(tmp_grid['xu'].values, tmp_grid['yu'].values),
                    crs=self.out_epsg)
                tmp_grid["geometry"] = tmp_grid["geometry"].apply(lambda gdf: [gdf])
                tmp_grid["ice_conc"] = sic_product.interp_ice_concentration(
                    sic_product.ice_conc, tmp_grid['xu'].values, tmp_grid['yu'].values)
                tmp_grid[self.target_var+'_drift_unc'] = 0.0
                tmp_grid['divergence'], tmp_grid['shear'] = [[0]] * len(tmp_grid), [[0]] * len(tmp_grid)
                self.master[beam][self.i][0] = tmp_grid
                self.scheme[(beams == beam).argmax(), self.i, 0] = 1

        elif self.sensor in ['cryosat2', 'sentinel3a', 'sentinel3b', 'envisat']:
            tmp_grid = gridding_lib.grid_data(sit, self.grid, [self.target_var], [self.target_var],
                                              hist_n_bins=hist_n_bins, hist_range=hist_range,
                                              agg_mode=['mean', 'std', 'hist'])
            unc_grid = gridding_lib.grid_data(sit, self.grid, [self.target_var + '_l2_unc'],
                                              [self.target_var + '_l2_unc'], agg_mode=['sum', 'cnt'])
            add_grid = gridding_lib.grid_data(sit, self.grid, self.add_variable+['time'],
                                              self.add_variable+['time'], agg_mode=['mean'])

            tmp_grid[self.target_var+'_l2_unc'] = np.sqrt(unc_grid[self.target_var+'_l2_unc_sum'])/unc_grid[
                self.target_var+'_l2_unc_cnt']
            tmp_grid[self.add_variable] = add_grid[self.add_variable]
            tmp_grid['t0'] = add_grid['time']
            tmp_grid['xu'] = tmp_grid.index.get_level_values('x')
            tmp_grid['yu'] = tmp_grid.index.get_level_values('y')
            tmp_grid['dt_days'] = 0
            tmp_grid.reset_index(drop=True, inplace=True)
            tmp_grid = gpd.GeoDataFrame(
                tmp_grid, geometry=gpd.points_from_xy(tmp_grid['xu'].values, tmp_grid['yu'].values), crs=self.out_epsg)
            tmp_grid["geometry"] = tmp_grid["geometry"].apply(lambda gdf: [gdf])
            tmp_grid["ice_conc"] = sic_product.interp_ice_concentration(
                sic_product.ice_conc, tmp_grid['xu'].values, tmp_grid['yu'].values)
            tmp_grid[self.target_var+'_drift_unc'] = 0.0
            tmp_grid['divergence'], tmp_grid['shear'] = [[0]] * len(tmp_grid), [[0]] * len(tmp_grid)
            self.master[self.i][0] = tmp_grid
            self.scheme[self.i, 0] = 1

        else:
            logger.error('Sensor does not exist: %s', self.sensor)
            sys.exit()

    def apply_drift_correction(self, j, tmp_grid, sid_product, sic_product, direct):
        # applies drift correction per day (24 h)
        dx, dy, dx_dy_unc = sid_product.drift_correction(tmp_grid['xu'].values, tmp_grid['yu'].values)
        div, she = sid_product.deformation(tmp_grid['xu'].values, tmp_grid['yu'].values)
        dt = np.full(len(dx), 24)
        dt_corr = 0
        if tmp_grid['dt_days'][0] == 0:
            dt_corr = (tmp_grid['t0'] - sid_product.ice_drift['time_bnds'][0])
            dt_corr = dt_corr / datetime.timedelta(days=1).total_seconds()

        if direct == 1:
            dt = dt - dt_corr
            xu = tmp_grid['xu'].values + (dx * dt)
            yu = tmp_grid['yu'].values + (dy * dt)
            tt = self.i + direct - j
        else:
            dt = dt + dt_corr
            xu = tmp_grid['xu'].values - (dx * dt)
            yu = tmp_grid['yu'].values - (dy * dt)
            tt = self.i + direct + j

        new_geom = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(xu, yu), crs=self.out_epsg)["geometry"].apply(lambda gdf: [gdf])
        tmp_grid["geometry"] = tmp_grid["geometry"] + new_geom
        tmp_grid['xu'], tmp_grid['yu'] = xu, yu
        tmp_grid[self.target_var+'_drift_unc'] += (dx_dy_unc * dt)**2
        tmp_grid['dt_days'] = tt - (self.i + direct)
        tmp_grid['divergence'] = tmp_grid.apply(lambda row: row['divergence'] + [div[row.name]], axis=1)
        tmp_grid['shear'] = tmp_grid.apply(lambda row: row['shear'] + [she[row.name]], axis=1)
        tmp_grid["ice_conc"] = sic_product.interp_ice_concentration(
            sic_product.ice_conc_ahead, tmp_grid['xu'].values, tmp_grid['yu'].values)
        tmp_grid = tmp_grid[tmp_grid["ice_conc"] > 0.15].reset_index(drop=True)
        return tmp_grid

    def drift_aware_proc(self, sid_product, sic_product, t_window_length, direct, day0):
        # incrementally applies drift correction and adds the corrected field to the master structure
        if self.sensor == 'icesat2':
            beams = np.array(['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r'])
            m = 0
            for beam in beams.tolist():
                m = 0
                end = self.i + 2 if direct == 1 else day0 - self.i + 2
                for j in range(1, end):
                    if j >= t_window_length:
                        continue
                    m = m + 1
                    if len(self.master[beam][self.i][(j - 1)]) == 0:
                        continue
                    tmp_grid = self.master[beam][self.i][(j - 1)].copy().reset_index(drop=True)
                    tmp_grid = self.apply_drift_correction(j, tmp_grid, sid_product, sic_product, direct)
                    self.master[beam][self.i + direct][j] = tmp_grid
                    self.scheme[(beams == beam).argmax(), self.i + direct, j] = 1

        elif self.sensor in ['cryosat2', 'sentinel3a', 'sentinel3b', 'envisat']:
            m = 0
            end = self.i + 2 if direct == 1 else day0 - self.i + 2
            for j in range(1, end):
                if j >= t_window_length:
                    continue
                m = m + 1
                if len(self.master[self.i][(j - 1)]) == 0:
                    continue
                tmp_grid = self.master[self.i][(j - 1)].copy().reset_index(drop=True)
                tmp_grid = self.apply_drift_correction(j, tmp_grid, sid_product, sic_product, direct)
                self.master[(self.i + direct)][j] = tmp_grid
                self.scheme[self.i + direct, j] = 1
        else:
            logger.error('Sensor does not exist: %s', self.sensor)
            sys.exit()

        return m

    def concat_gdfs(self, gdf_array_index, row_lim):
        gdf_list = []
        for j in range(0, row_lim + 1):
            if self.sensor == 'icesat2':
                beams = np.array(['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r'])
                for beam in beams.tolist():
                    if len(self.master[beam][gdf_array_index][j]) != 0:
                        gdf_list.append(self.master[beam][gdf_array_index][j])
                        del self.master[beam][gdf_array_index][j]
            elif self.sensor in ['cryosat2', 'sentinel3a', 'sentinel3b', 'envisat']:
                if len(self.master[gdf_array_index][j]) != 0:
                    gdf_list.append(self.master[gdf_array_index][j])
                    del self.master[gdf_array_index][j]
        return pd.concat(gdf_list).reset_index(drop=True)
        # return pd.concat(gdf_list).pipe(gpd.GeoDataFrame, crs=self.out_epsg).reset_index(drop=True)
