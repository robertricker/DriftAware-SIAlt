from driftaware_sialt.gridding import gridding_lib
import pandas as pd
import geopandas as gpd
import numpy as np
import sys
from loguru import logger
import datetime
from numbers import Real
from driftaware_sialt.filters.marginal_ice_zone import compute_apply_flag


def grid_parcel_uncertainty(data, grid, source_unc_var, parcel_unc_var):
    """Propagate source uncertainties to the mean value of each parcel."""
    uncertainty_variance = data[[source_unc_var, 'geometry']].copy()
    uncertainty_variance[source_unc_var] **= 2
    uncertainty_grid = gridding_lib.grid_data(
        uncertainty_variance,
        grid,
        [source_unc_var],
        [parcel_unc_var],
        agg_mode=['sum', 'cnt'])
    return (
        np.sqrt(uncertainty_grid[parcel_unc_var + '_sum'])
        / uncertainty_grid[parcel_unc_var + '_cnt'])


def grid_mission_fractions(data, grid, configured_sensors, available_sensors):
    """Grid mission indicators for the sensors available on the current date.
    Keep columns for every configured sensor in the result so downstream output
    retains a stable schema.  Sensors without input data are represented by
    zero fractions and counts.
    """
    available_sensors = [
        sensor for sensor in configured_sensors
        if sensor in available_sensors and sensor in data.columns
    ]
    if not available_sensors:
        raise ValueError('No available sensor columns found in altimetry data')

    mission_grid = gridding_lib.grid_data(
        data, grid, available_sensors, available_sensors,
        agg_mode=['mean', 'sum'])
    for sensor in configured_sensors:
        if sensor not in available_sensors:
            mission_grid[sensor] = 0.0
            mission_grid[sensor + '_sum'] = 0.0
    return mission_grid


def ocean_heat_flux_at_points(source, x, y):
    """Return constant or spatially interpolated ocean heat flux."""
    if isinstance(source, Real):
        return np.full(len(x), float(source))

    values = np.asarray(source.interp_ocean_heat_flux(x, y)).reshape(-1)
    if len(values) != len(x):
        raise ValueError(
            'Interpolated ocean heat flux does not match parcel count')
    return values


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

    def baseline_proc(
            self, sic_product, hist_n_bins, hist_range, sit_clim=None,
            thermo_model=None, available_sensors=None):
        # adds the original measurements at t=0 (without drift correction) to the master structure
        sit = self.parent.product
        if available_sensors is None:
            available_sensors = [
                sensor for sensor in self.sensor if sensor in sit.columns
            ]
        source_unc_var = self.target_var + '_l2_unc'
        parcel_unc_var = self.target_var + '_parcel_unc'
        target_sensors = ['cryosat2', 'sentinel3a', 'sentinel3b', 'envisat']
        
        if sit_clim is not None:
            sit = compute_apply_flag(sit, sic_product, sit_clim, self.target_var, available_sensors, crs=self.out_epsg)
           
        if 'icesat2' in available_sensors:
            beams = np.array(['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r'])
            for beam in sit.beam.unique(): 
                tmp = (sit[[self.target_var, source_unc_var, 'geometry', 'time', 'beam'] + self.add_variable]
                           .copy()
                           .loc[sit['beam'] == beam]
                           .drop(columns=['beam'])
                           )
                tmp = tmp.reset_index(drop=True)
                tmp_grid = gridding_lib.grid_data(tmp, self.grid, [self.target_var], [self.target_var],
                                                  hist_n_bins=hist_n_bins, hist_range=hist_range,
                                                  agg_mode=['mean', 'std', 'hist'])
                add_grid = gridding_lib.grid_data(tmp, self.grid, self.add_variable+['time'],
                                                  self.add_variable+['time'], agg_mode=['mean'])
                

                tmp_grid[parcel_unc_var] = grid_parcel_uncertainty(
                    tmp, self.grid, source_unc_var, parcel_unc_var)
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
                tmp_grid['_ice_conc_products'] = sic_product.product_id
                tmp_grid['_ice_drift_products'] = ''
                tmp_grid[self.target_var+'_drift_unc'] = 0.0
                tmp_grid['divergence'], tmp_grid['shear'] = [[0]] * len(tmp_grid), [[0]] * len(tmp_grid)
                
                self.master[beam][self.i][0] = tmp_grid
                self.scheme[(beams == beam).argmax(), self.i, 0] = 1

        elif any(s in available_sensors for s in target_sensors):
            tmp_grid = gridding_lib.grid_data(sit, self.grid, [self.target_var], [self.target_var],
                                              hist_n_bins=hist_n_bins, hist_range=hist_range,
                                              agg_mode=['mean', 'std', 'hist'])
            add_grid = gridding_lib.grid_data(sit, self.grid, self.add_variable+['time'],
                                              self.add_variable+['time'], agg_mode=['mean'])
            frac_mission_grid = grid_mission_fractions(
                sit, self.grid, self.sensor, available_sensors)

            tmp_grid[parcel_unc_var] = grid_parcel_uncertainty(
                sit, self.grid, source_unc_var, parcel_unc_var)
  
            tmp_grid[self.add_variable] = add_grid[self.add_variable]
            tmp_grid[self.sensor] = frac_mission_grid[self.sensor]
            tmp_grid[[s + '_cnt' for s in self.sensor]] = frac_mission_grid[[s + '_sum' for s in self.sensor]]
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
            tmp_grid['_ice_conc_products'] = sic_product.product_id
            tmp_grid['_ice_drift_products'] = ''
            tmp_grid[self.target_var+'_drift_unc'] = 0.0
            tmp_grid['divergence'], tmp_grid['shear'] = [[0]] * len(tmp_grid), [[0]] * len(tmp_grid)
            if thermo_model:
                tmp_grid['sit_corr_thermo_mod'] = tmp_grid['sea_ice_thickness']
                tmp_grid['thermo_change_mod'] = [0] * len(tmp_grid)
                tmp_grid['thermo_growth_mod'] = [0] * len(tmp_grid)
                tmp_grid['t2m'] = [0] * len(tmp_grid)
                tmp_grid['ohf'] = [0] * len(tmp_grid)

            self.master[self.i][0] = tmp_grid
            self.scheme[self.i, 0] = 1  

        else:
            logger.error('Sensor does not exist: %s', self.sensor)
            sys.exit()

    def apply_drift_correction(self, j, tmp_grid, sid_product, sic_product, t2m_product, ohf_product, direct, thermo_model=None):
        # applies drift correction per day (24 h)
        dx, dy, dx_dy_unc = sid_product.drift_correction(tmp_grid['xu'].values, tmp_grid['yu'].values)

        div, she = sid_product.deformation(tmp_grid['xu'].values, tmp_grid['yu'].values)
        
        if thermo_model:
            tmp_grid['ohf'] = ocean_heat_flux_at_points(
                ohf_product,
                tmp_grid['xu'].values,
                tmp_grid['yu'].values)
            tmp_grid['t2m'], thermodyn_growth, thermodyn_corr_sit = t2m_product.thermodyn_growth(thermo_model, tmp_grid['sit_corr_thermo_mod'], tmp_grid['snow_depth'],
                                                tmp_grid['xu'].values, tmp_grid['yu'].values, direct, tmp_grid['ohf'])
            
        dt = np.full(len(dx), 24)
        dt_corr = 0
        if tmp_grid['dt_days'][0] == 0:
            dt_corr = (tmp_grid['t0'] - sid_product.ice_drift['time_bnds'][0])
            dt_corr = dt_corr / datetime.timedelta(hours=1).total_seconds()

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
        if thermo_model:
            tmp_grid['sit_corr_thermo_mod'] = tmp_grid.apply(lambda row: thermodyn_corr_sit[row.name], axis=1)
            # what is called growth is the same as the computed growth (always in the time direction even for backward drifting)
            tmp_grid['thermo_growth_mod'] = tmp_grid.apply(lambda row: row['thermo_growth_mod'] + thermodyn_growth[row.name], axis=1)
            #refer to the deltaH that need the sit needs to be corrected from
            tmp_grid['thermo_change_mod'] = tmp_grid.apply(lambda row: row['thermo_change_mod'] + thermodyn_growth[row.name]*direct, axis=1) 
        tmp_grid["ice_conc"] = sic_product.interp_ice_concentration(
            sic_product.ice_conc_ahead, tmp_grid['xu'].values, tmp_grid['yu'].values)
        tmp_grid['_ice_conc_products'] += '|' + sic_product.product_id
        tmp_grid['_ice_drift_products'] += '|' + sid_product.product_id
        tmp_grid = tmp_grid[tmp_grid["ice_conc"] > 0.15].reset_index(drop=True)
        return tmp_grid

    def drift_aware_proc(self, sid_product, sic_product, t2m_product, ohf_product, t_window_length, direct, day0, thermo_model):
        target_sensors = ['cryosat2', 'sentinel3a', 'sentinel3b', 'envisat']

        if 'icesat2' in self.sensor:
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
                    tmp_grid = self.apply_drift_correction(j, tmp_grid, sid_product, sic_product, t2m_product, ohf_product, direct, thermo_model=None)
                    self.master[beam][self.i + direct][j] = tmp_grid
                    self.scheme[(beams == beam).argmax(), self.i + direct, j] = 1

        elif any(s in self.sensor for s in target_sensors):
            m = 0
            # end is the indice of the target day +2 if f or -2 if r
            end = self.i + 2 if direct == 1 else day0 - self.i + 2  
            # go from 1 to the end of the curent indice + 1
            # j is the lag in data acquisition
            for j in range(1, end): 
                
                if j >= t_window_length:
                    continue
                m = m + 1
                if len(self.master[self.i][(j - 1)]) == 0:
                    continue
                tmp_grid = self.master[self.i][(j - 1)].copy().reset_index(drop=True)
                tmp_grid = self.apply_drift_correction(j, tmp_grid, sid_product, sic_product, t2m_product, ohf_product, direct, thermo_model=thermo_model)
                self.master[(self.i + direct)][j] = tmp_grid
                self.scheme[self.i + direct, j] = 1
        else:
            logger.error('Sensor does not exist: %s', self.sensor)
            sys.exit()

        return m

    def concat_gdfs(self, gdf_array_index, row_lim):
        gdf_list = []
        target_sensors = ['cryosat2', 'sentinel3a', 'sentinel3b', 'envisat']
        for j in range(0, row_lim + 1):
            if 'icesat2' in self.sensor:
                beams = np.array(['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r'])
                for beam in beams.tolist():
                    if len(self.master[beam][gdf_array_index][j]) != 0:
                        gdf_list.append(self.master[beam][gdf_array_index][j])
                        del self.master[beam][gdf_array_index][j]
            elif any(s in self.sensor for s in target_sensors) :
                if len(self.master[gdf_array_index][j]) != 0:
                    gdf_list.append(self.master[gdf_array_index][j])
                    del self.master[gdf_array_index][j]
        return pd.concat(gdf_list).reset_index(drop=True)
