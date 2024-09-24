from loguru import logger
import pyproj
import numpy as np
import sys


class PrepareNetcdf:
    def __init__(self, config):
        self.target_var = config["options"]["target_variable"]
        self.hist_n_bins = config['options']['proc_step_options']['stacking']['hist']['n_bins']
        self.hist_range = config['options']['proc_step_options']['stacking']['hist']['range'][
            "freeboard" if "freeboard" in self.target_var else "thickness"]
        self.mode = config['options']['proc_step_options']['gridding']["mode"]
        self.crs = pyproj.CRS.from_epsg(int(config['options']['out_epsg'].split(":")[1]))
        self.sid_product = config['options']['ice_conc_product']
        self.sic_product = config['options']['ice_drift_product']

    def set_variables(self):
        if self.mode == 'drift-aware':
            var = [self.target_var, 'dist_acquisition', 'dt_days', 'ice_conc', 'deformation', 'shear', 'divergence',
                   'snow_depth', 'drift_unc',
                   self.target_var + '_l2_unc', self.target_var + '_drift_unc',
                   self.target_var + '_growth_unc', self.target_var + '_total_unc',
                   self.target_var + '_mode', self.target_var + '_corr', 'growth', 'growth_interpolated']

            var_rename = [self.target_var, 'dist_acquisition', 'time_offset_acquisition', 'sea_ice_concentration',
                          'deformation', 'shear', 'divergence', 'snow_depth', 'drift_unc',
                          self.target_var + '_l2_unc', self.target_var+'_drift_unc',
                          self.target_var+'_growth_unc', self.target_var+'_total_unc',
                          self.target_var+'_mode', self.target_var+'_corrected',
                          self.target_var+'_growth', self.target_var+'_growth_interpolated']

        elif self.mode == 'conventional':
            var = [self.target_var, 'dist_acquisition', 'dt_days', 'ice_conc', 'deformation',  'shear', 'divergence',
                   'snow_depth',
                   self.target_var + '_l2_unc', self.target_var + '_mode']
            var_rename = [self.target_var, 'dist_acquisition', 'time_offset_acquisition',
                          'sea_ice_concentration', 'deformation',  'shear', 'divergence', 'snow_depth',
                          self.target_var + '_l2_unc', self.target_var+'_mode']
        else:
            logger.error('Gridding mode does not exist: %s', self.mode)
            sys.exit()
        return var, var_rename

    def add_projection_field(self, xarray):
        proj_name = self.crs.name.split('/', 1)[-1].strip().replace(" ", "_")
        xarray[proj_name] = np.iinfo(np.int32).min
        xarray[proj_name].attrs = {'long_name': proj_name.replace("_", " "),
                                   'grid_mapping_name': 'lambert_azimuthal_equal_area',
                                   'false_easting': 0.0,
                                   'false_northing': 0.0,
                                   'latitude_of_projection_origin': 90.0,
                                   'longitude_of_projection_origin': 0.0,
                                   'longitude_of_prime_meridian': 0.0,
                                   'semi_major_axis': 6378137.0,
                                   'inverse_flattening': 298.257223563,
                                   'coordinates': '(latitude, longitude)',
                                   'units': 'm',
                                   'proj4_string': self.crs.to_proj4()}
        return xarray

    def drop_fields(self, master):
        if self.mode == 'drift-aware':
            master.drop(columns=['dist_acquisition_std', 'time_offset_acquisition_std', 'sea_ice_concentration_std',
                                 'deformation_std', 'shear_std', 'divergence_std', 'snow_depth_std', 'drift_unc_std',
                                 self.target_var + '_mode_std', self.target_var + '_corrected_std',
                                 self.target_var + '_l2_unc_std', self.target_var+'_drift_unc_std',
                                 self.target_var+'_growth_unc_std', self.target_var+'_total_unc_std',
                                 self.target_var + '_growth_std',
                                 self.target_var + '_growth_interpolated_std'], inplace=True)
        elif self.mode == 'conventional':
            master.drop(columns=['dist_acquisition_std', 'time_offset_acquisition_std', 'sea_ice_concentration_std',
                                 'deformation_std', 'shear_std', 'divergence_std', 'snow_depth_std',
                                 self.target_var + '_l2_unc_std',
                                 self.target_var + '_mode_std'], inplace=True)
        else:
            logger.error('Gridding mode does not exist: %s', self.mode)
            sys.exit()
        return master

    def set_glob_attrbs(self, xarray):
        xarray.attrs['Description'] = 'Drift-aware ' + self.target_var.replace("_", " ") +\
                                      ' using low resolution sea ice drift, sea ice concentration,' \
                                      ' and sea ice thickness along track data (trajectories)'
        xarray.attrs['sea ice drift product'] = self.sid_product
        xarray.attrs['sea ice concentration product'] = self.sic_product

        xarray.attrs['Contact'] = 'rori@norceresearch.no'
        xarray.attrs['Reference'] = 'TBD'
        xarray.attrs['Conventions'] = "CF-1.8"

        return xarray

    def set_var_attrbs(self, xarray):
        if self.target_var == "sea_ice_freeboard":
            long_name = {"target_var": "sea ice elevation above sea level",
                         "target_var_unc": "sea ice freeboard uncertainty from level2 product",
                         "target_var_drift_unc": "sea ice freeboard uncertainty from drift correction",
                         "target_var_growth_unc": "sea ice freeboard uncertainty from growth correction",
                         "target_var_total_unc": "combined sea ice freeboard uncertainty",
                         "target_var_std": "standard deviation of binned sea ice freeboard estimates",
                         "target_var_mode": "modal sea ice freeboard",
                         "target_var_corrected": "growth-corrected sea ice freeboard",
                         "target_var_growth": "sea ice freeboard growth within t_window_length",
                         "target_var_growth_interp": "interpolated sea ice freeboard growth within t_window_length"
                         }
        elif self.target_var == "total_freeboard":
            long_name = {"target_var": "sea ice + snow elevation above sea level",
                         "target_var_unc": "total freeboard uncertainty from level2 product",
                         "target_var_drift_unc": "total freeboard uncertainty from drift correction",
                         "target_var_growth_unc": "total freeboard uncertainty from growth correction",
                         "target_var_total_unc": "combined total freeboard uncertainty",
                         "target_var_std": "standard deviation of binned total freeboard estimates",
                         "target_var_mode": "modal total freeboard",
                         "target_var_corrected": "growth-corrected total freeboard",
                         "target_var_growth": "total freeboard growth within t_window_length",
                         "target_var_growth_interp": "interpolated total freeboard growth within t_window_length"
                         }
        elif self.target_var == "sea_ice_thickness":
            long_name = {"target_var": "thickness of the sea ice layer",
                         "target_var_unc": "sea ice thickness uncertainty from level2 product",
                         "target_var_drift_unc": "sea ice thickness uncertainty from drift correction",
                         "target_var_growth_unc": "sea ice thickness uncertainty from growth correction",
                         "target_var_total_unc": "combined sea ice thickness uncertainty",
                         "target_var_std": "standard deviation of binned sea ice thickness estimates",
                         "target_var_mode": "modal sea ice thickness",
                         "target_var_corrected": "growth-corrected sea ice thickness",
                         "target_var_growth": "sea ice thickness growth within t_window_length",
                         "target_var_growth_interp": "interpolated sea ice thickness growth within t_window_length"
                         }
        else:
            logger.error('Target variable does not exist: %s', self.target_var)
            sys.exit()

        hist_bin_size = (self.hist_range[1]-self.hist_range[0])/self.hist_n_bins

        xarray[self.target_var].attrs = {'long_name': long_name["target_var"], 'units': 'm'}
        xarray[self.target_var+"_l2_unc"].attrs = {'long_name': long_name["target_var_unc"], 'units': 'm'}
        xarray[self.target_var+'_std'].attrs = {'long_name': long_name["target_var_std"], 'units': 'm'}
        xarray[self.target_var+'_mode'].attrs = {'long_name': long_name["target_var_mode"], 'units': 'm'}
        xarray[self.target_var + '_hist'].attrs = {
            'long_name': (
                    long_name["target_var"] + ' classes between ' +
                    str(self.hist_range[0]) + ' m and ' +
                    str(self.hist_range[1]) + ' m with ' +
                    str(hist_bin_size) + 'm bin size'
            ),
            'units': 'counts'
        }

        if self.mode == 'drift-aware':
            xarray[self.target_var+"_corrected"].attrs = {'long_name': long_name["target_var_corrected"], 'units': 'm'}
            xarray[self.target_var+"_growth"].attrs = {
                'long_name': long_name["target_var_growth"], 'units': 'm/day'}
            xarray[self.target_var + "_growth_unc"].attrs = {'long_name': long_name["target_var_growth_unc"],
                                                             'units': 'm'}
            xarray[self.target_var+"_growth_interpolated"].attrs = {
                'long_name': long_name["target_var_growth_interp"], 'units': 'm/day'}
            xarray[self.target_var + "_drift_unc"].attrs = {'long_name': long_name["target_var_drift_unc"],
                                                            'units': 'm'}
            xarray[self.target_var + "_total_unc"].attrs = {'long_name': long_name["target_var_total_unc"],
                                                            'units': 'm'}
            xarray.drift_unc.attrs = {'long_name': 'drift uncertainty',
                                      'units': 'm'}

        xarray.time.attrs = {'long_name': 'reference time of product',
                             'axis': "T",
                             'units': 'seconds since 1970-01-01 00:00:00'}

        xarray.time_bnds.attrs = {'long_name': 'period over which ' + self.target_var.replace("_", " ") +
                                               ' data have been collected',
                                  'units': 'seconds since 1970-01-01 00:00:00'}

        xarray.xc.attrs = {'long_name': 'x coordinate of projection (eastings)',
                           'units': 'm'}

        xarray.yc.attrs = {'long_name': 'y coordinate of projection (northings)',
                           'units': 'm'}

        xarray.zc.attrs = {'long_name': 'z coordinate for histogram (vertical)',
                           'units': 'm'}

        xarray.longitude.attrs = {'long_name': 'longitude coordinate',
                                  'units': 'degrees_east'}

        xarray.latitude.attrs = {'long_name': 'latitude coordinate',
                                 'units': 'degrees_north'}

        xarray.dist_acquisition.attrs = {'long_name': 'distance from the location of data aquisition',
                                         'units': 'km'}

        xarray.time_offset_acquisition.attrs = {'long_name': 'time offfset to data aquisition',
                                                'units': 'days'}

        xarray.sea_ice_concentration.attrs = {'long_name': 'sea ice concentraion',
                                              'units': 'percentage'}

        xarray.deformation.attrs = {'long_name': 'mean deformation accumulated along the trajectory',
                                    'units': '1/day'}

        xarray.shear.attrs = {'long_name': 'mean shear accumulated along the trajectory',
                                    'units': '1/day'}

        xarray.divergence.attrs = {'long_name': 'mean divergence accumulated along the trajectory',
                                    'units': '1/day'}

        xarray.snow_depth.attrs = {'long_name': 'snow depth',
                                    'units': 'm'}

        xarray.region_flag.attrs = {'long_name': 'NSIDC region mask v2',
                                    'description': "National Snow and Ice Data Center (NSIDC) Northern Hemisphere"
                                                   " region mask (preliminary) courtesy W. Meier and S. Stewart, NSIDC",
                                    'flags': "0: Outside of defined regions, 1: Central Arctic, 2: Beaufort Sea,"
                                             " 3: Chukchi Sea, 4: East Siberian Sea, 5: Laptev Sea, 6: Kara Sea,"
                                             " 7: Barents Sea, 8: East Greenland Sea, 9: Baffin Bay and Davis Strait,"
                                             " 10: Gulf of St. Lawrence, 11: Hudson Bay, 12: Canadian Archipelago,"
                                             " 13: Bering Sea, 14: Sea of Okhotsk, 15: Sea of Japan, 16: Bohai Sea,"
                                             " 17: Gulf of Bothnia, Baltic Sea, 18: Gulf of Alaska"
                                    }
        return xarray
