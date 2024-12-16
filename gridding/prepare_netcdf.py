from loguru import logger
import pyproj
import numpy as np
import xarray as xr
import sys
import os
import re
import yaml
import datetime
from jinja2 import Template
from shapely.geometry import Point


class PrepareNetcdf:
    def __init__(self, config, file, region_grid):
        with open(os.path.join(os.path.dirname(__file__), 'netcdf_config.yaml'), 'r') as f:
            self.netcdf_config = yaml.safe_load(f)
        self.target_var = config["options"]["target_variable"]
        self.hist_n_bins = config['options']['proc_step_options']['stacking']['hist']['n_bins']
        self.hist_range = config['options']['proc_step_options']['stacking']['hist']['range'][
            "freeboard" if "freeboard" in self.target_var else "thickness"]
        self.mode = config['options']['proc_step_options']['gridding']["mode"]
        self.crs = pyproj.CRS.from_epsg(int(config['options']['out_epsg'].split(":")[1]))
        self.sid_product = config['options']['ice_conc_product']
        self.sic_product = config['options']['ice_drift_product']
        self.out_epsg = config["options"]["out_epsg"]
        self.hem = config["options"]["hemisphere"]
        self.version = config['version']
        self.file = file
        self.region_grid = region_grid

    def make_netcdf_filename(self, grid, gridding_mode):
        centr = grid['geometry'][0].centroid
        dist = [centr.distance(Point(vertex)) for vertex in list(grid['geometry'][0].exterior.coords)]
        dist_km = "{:.0f}".format(round(min(dist) * np.sqrt(2)) / 1e3)
        prefix = '-'.join(re.split('-', os.path.basename(self.file))[:2])
        prdlvl = 'L3C'
        var = re.split('-', os.path.basename(self.file))[3]
        instr = re.split('-', os.path.basename(self.file))[4]
        proj_map = {
            "EPSG:6931": "EASE2",
            "EPSG:6932": "EASE2"}
        extra = (f"{self.hem.upper()}_"
                 f"{dist_km}KM_{proj_map.get(self.out_epsg)}_{gridding_mode.upper()}")
        period = re.split('-', os.path.basename(self.file))[6]
        version = re.search(r'(fv\d+)', os.path.basename(self.file)).group(1)
        return f"{prefix}-{prdlvl}-{var}-{instr}-{extra}-{period}-{version}.nc"

    def select_variables(self):
        var = [Template(item).render(target_var=self.target_var)
               for item in self.netcdf_config['variables'][self.mode]['include']]
        var_rename = [Template(item).render(target_var=self.target_var)
                      for item in self.netcdf_config['variables'][self.mode]['rename']]
        return var, var_rename

    def set_var_attrbs(self, dataset):
        variable_attributes = self.netcdf_config['variable_attributes']
        for var_name, attributes in variable_attributes.items():
            if var_name in dataset:
                dataset[var_name].attrs.update(attributes)
        return dataset

    def add_histogram(self, xarray):
        if self.netcdf_config['variables'][self.mode]['histogram']:
            bin_size = (self.hist_range[1] - self.hist_range[0]) / self.hist_n_bins
            hist_bins = np.arange(self.hist_range[0] + bin_size / 2, self.hist_range[1] + bin_size / 2, bin_size)
            hist_arr = xr.concat([xarray[str(i) + '_sum'] for i in range(self.hist_n_bins)], dim='hist_bins').values
            xarray = xarray.assign_coords(hist_bins=("hist_bins", hist_bins))
            xarray[self.target_var + '_hist'] = (['time', 'yc', 'xc', 'hist_bins'],
                                                 np.transpose(hist_arr, (1, 2, 3, 0)))
        xarray = xarray.drop_vars([str(i) + '_sum' for i in range(self.hist_n_bins)])
        return xarray

    def add_region_flag(self, xarray):
        xarray['region_flag'] = (['time', 'yc', 'xc'], np.flip(self.region_grid, axis=0)[np.newaxis, :, :])
        xarray['region_flag'].attrs = {'standard_name': 'region_flag',
                                       'long_name': 'NSIDC region mask v2',
                                       'coordinates': 'time longitude latitude',
                                       'flag_meanings': 'undefined_region central_arctic beaufort_sea chukchi_sea '
                                                        'east_siberian_sea laptev_sea kara_sea barents_sea '
                                                        'east_greenland_sea baffin_bay_and_davis_strait '
                                                        'gulf_of_st_lawrence hudson_bay canadian_archipelago '
                                                        'bering_sea sea_of_okhotsk sea_of_japan bohai_sea baltic_sea '
                                                        'gulf_of_alaska',
                                       'flag_values': np.byte(np.arange(19)),
                                       'units': '1',
                                       'grid_mapping': 'crs',
                                       'comment': ""}
        return xarray

    @staticmethod
    def add_time_bnds(xarray, t0, t1):
        time_bnds = np.array([[t0, t1]])
        xarray["time_bnds"] = xr.DataArray(time_bnds, dims=("time", "nv"), coords={"time": xarray.time})
        return xarray

    def add_projection_field(self, xarray):
        proj_name = 'crs'  # self.crs.name.split('/', 1)[-1].strip().replace(" ", "_")
        xarray[proj_name] = np.iinfo(np.int32).min
        xarray[proj_name].attrs = {'long_name': self.crs.name.split('/', 1)[-1].strip(),
                                   'grid_mapping_name': 'lambert_azimuthal_equal_area',
                                   'false_easting': 0.0,
                                   'false_northing': 0.0,
                                   'latitude_of_projection_origin':
                                       float(pyproj.CRS.from_string(self.out_epsg ).to_dict().get("lat_0")),
                                   'longitude_of_projection_origin': 0.0,
                                   'longitude_of_prime_meridian': 0.0,
                                   'proj4_string': self.crs.to_proj4()}
        return xarray

    def set_glob_attrbs(self, xarray):
        t0, t1 = np.array(xarray['time_bnds'][0])[0], np.array(xarray['time_bnds'][0])[1]
        xarray.attrs['summary'] = 'Drift-aware ' + self.target_var.replace("_", " ") +\
                                  ' using low resolution sea ice drift, sea ice concentration,' \
                                  ' and sea ice thickness along track data (trajectories)'
        xarray.attrs['title'] = 'Drift-aware sea ice thickness'
        xarray.attrs['institution'] = 'NORCE'
        xarray.attrs['comment'] = 'These data were produced by NORCE'
        xarray.attrs['contact_email'] = 'rori@norceresearch.no'
        xarray.attrs['source'] = f"{self.sid_product}, {self.sic_product}, ESA-CCI L2P-SIT v3.0"
        xarray.attrs['product_version'] = self.version
        xarray.attrs['project'] = 'ESA CCI'
        xarray.attrs['geospatial_lat_min'] = np.min(np.array(xarray['latitude'][0]))
        xarray.attrs['geospatial_lat_max'] = np.max(np.array(xarray['latitude'][0]))
        xarray.attrs['geospatial_lon_min'] = np.min(np.array(xarray['longitude'][0]))
        xarray.attrs['geospatial_lon_max'] = np.max(np.array(xarray['longitude'][0]))
        xarray.attrs['time_coverage_start'] = datetime.datetime.utcfromtimestamp(t0).strftime('%Y%m%d')+"T000000Z"
        xarray.attrs['time_coverage_end'] = datetime.datetime.utcfromtimestamp(t1).strftime('%Y%m%d')+"T000000Z"
        xarray.attrs['license'] = "ESA CCI Data Policy: free and open access"
        xarray.attrs['production_date'] = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

        return xarray
