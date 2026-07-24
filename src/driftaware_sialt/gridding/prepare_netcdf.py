import pyproj
import numpy as np
import xarray as xr
import os
import yaml
import datetime
from jinja2 import Template
from shapely.geometry import Point


class PrepareNetcdf:
    def __init__(self, config, file, region_grid, region_metadata, source_stack,
                 source_products):
        with open(os.path.join(os.path.dirname(__file__), 'netcdf_config.yaml'), 'r') as f:
            self.netcdf_config = yaml.safe_load(f)
        
        self.sensor = config['options']['sensor']
        self.target_var = config["options"]["target_variable"]
        self.add_variables = config['options'].get('add_variable',
                                                  config['gridding'].get('add_variables', []))
        self.hist_n_bins = source_stack['histogram']['n_bins']
        self.hist_range = source_stack['histogram']['range'][
            "freeboard" if "freeboard" in self.target_var else "thickness"]
        self.mode = config['gridding']["mode"]
        self.crs = pyproj.CRS.from_epsg(int(config['options']['out_epsg'].split(":")[1]))
        self.source_products = source_products
        self.out_epsg = config["options"]["out_epsg"]
        self.hem = config["options"]["hemisphere"]
        self.version = config['version']
        self.file = file
        self.region_grid = region_grid
        self.region_metadata = region_metadata

    def make_netcdf_filename(self, grid, gridding_mode):
        centr = grid['geometry'][0].centroid
        dist = [centr.distance(Point(vertex)) for vertex in list(grid['geometry'][0].exterior.coords)]
        dist_km = "{:.0f}".format(round(min(dist) * np.sqrt(2)) / 1e3)
        filename_parts = os.path.splitext(os.path.basename(self.file))[0].split('-')
        try:
            product_level_index = filename_parts.index('L2P')
        except ValueError as error:
            raise ValueError(
                f"trajectory filename has no L2P product level: {self.file}") from error

        prefix = '-'.join(filename_parts[:product_level_index])
        prdlvl = 'L3C'
        source_fields = filename_parts[product_level_index + 1:]
        if len(source_fields) < 6 or not source_fields[0].startswith(('DA', 'CV')):
            raise ValueError(
                f"unexpected trajectory filename layout: {self.file}")
        var = source_fields[1]
        instr = source_fields[2]
        proj_map = {
            "EPSG:6931": "EASE2",
            "EPSG:6932": "EASE2"}
        extra = (f"{self.hem.upper()}_"
                 f"{dist_km}KM_{proj_map.get(self.out_epsg)}")
        period = filename_parts[-2]
        version = f"fv{self.version}"
        mode = gridding_mode.upper()
        return f"{prefix}-{prdlvl}-{mode}-{var}-{instr}-{extra}-{period}-{version}.nc"

    def select_variables(self, data):
        var = [Template(item).render(target_var=self.target_var) #Exception for is2
               for item in self.netcdf_config['variables'][self.mode]['include']
               if not (self.sensor == 'icesat2' and item == 'snow_depth') 
               and Template(item).render(target_var=self.target_var) in data.columns] 
                #if Template(item).render(target_var=self.target_var) in data.columns and item!='snow_depth']

        var_rename = []

        include_items = self.netcdf_config['variables'][self.mode]['include']
        rename_items = self.netcdf_config['variables'][self.mode]['rename']

        for i, item in enumerate(rename_items):
            if self.sensor == 'icesat2' and include_items[i] == 'snow_depth':
                continue  # exception spéciale
    
            input_var = Template(include_items[i]).render(target_var=self.target_var)
            output_var = Template(item).render(target_var=self.target_var)

            if input_var in data.columns:
                var_rename.append(output_var)
        #var_rename = [Template(item).render(target_var=self.target_var) #Same exception for is2
        #              for item in self.netcdf_config['variables'][self.mode]['rename']
        #              if not (self.sensor == 'icesat2' and item == 'snow_depth')
        #              and item in data.columns] 
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

    def add_region_code(self, xarray):
        flag_values = self.region_metadata['flag_values']
        region_code = np.flip(self.region_grid, axis=0).astype(flag_values.dtype)
        xarray['region_code'] = (
            ['time', 'yc', 'xc'], region_code[np.newaxis, :, :])
        xarray['region_code'].attrs = {
            'long_name': self.region_metadata['long_name'],
            'coordinates': 'time longitude latitude',
            'flag_meanings': self.region_metadata['flag_meanings'],
            'flag_values': flag_values,
            'units': '1',
            'grid_mapping': 'crs',
            'source': self.region_metadata['source'],
            'comment': ''}
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
                                       float(pyproj.CRS.from_string(self.out_epsg).to_dict().get("lat_0")),
                                   'longitude_of_projection_origin': 0.0,
                                   'longitude_of_prime_meridian': 0.0,
                                   'proj4_string': self.crs.to_proj4()}
        return xarray

    def set_glob_attrbs(self, xarray):
        t0, t1 = np.array(xarray['time_bnds'][0])[0], np.array(xarray['time_bnds'][0])[1]
        xarray.attrs['Conventions'] = 'CF-1.6'
        xarray.attrs['summary'] = 'Drift-aware ' + self.target_var.replace("_", " ") +\
                                  ' using low resolution sea ice drift, sea ice concentration,' \
                                  ' and sea ice thickness along track data (trajectories)'
        xarray.attrs['title'] = 'Drift-aware sea ice thickness'
        xarray.attrs['institution'] = 'NORCE'
        xarray.attrs['comment'] = 'These data were produced by NORCE'
        xarray.attrs['contact_email'] = 'rori@norceresearch.no'
        labels = {
            'sea_ice_concentration': 'sea ice concentration',
            'sea_ice_drift': 'sea ice drift'}
        sources = [
            f"{labels[key]}: {', '.join(self.source_products[key])}"
            for key in labels if self.source_products.get(key)]
        sources.append('ESA-CCI L2P-SIT v3.0')
        xarray.attrs['source'] = '; '.join(sources)
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
