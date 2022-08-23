import datetime
import stacking_lib
import gridding_lib
import pandas as pd
import geopandas as gpd
import shapely as shp
import numpy as np
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import griddata
import warnings
import cartopy.crs as ccrs
import cartopy
import glob
import logging
import sys
from datetime import date
import datetime
from multiprocessing.pool import Pool
from astropy.time import Time
from tqdm.notebook import tqdm
from pyproj import Proj, transform
import gps_time
import re
import json
import pickle
import statistics as stat
import sys
import os


def generate_grids(config):
    if config["options"]["sensor"] == 'is2':
        source_list = sorted(glob.glob(config["dir"]["is2_geojson"] + "/" + '*.geojson'))
        out_dir = config["dir"]["is2_netcdf"]
    else:
        source_list = sorted(glob.glob(config["dir"]["cs2_geojson"] + "/" + '*.geojson'))
        out_dir = config["dir"]["cs2_netcdf"]

    grid, cell_width = gridding_lib.define_grid(
        config["netcdf_grid"]["bounds"][0],
        config["netcdf_grid"]["bounds"][1],
        config["netcdf_grid"]["bounds"][2],
        config["netcdf_grid"]["bounds"][3],
        config["netcdf_grid"]["dim"],
        config["netcdf_grid"]["epsg"])

    for file in source_list:
        print(file)
        data = gpd.read_file(file)

        data['dist_acquisition'] = data.distance(
            gpd.GeoDataFrame(geometry=gpd.points_from_xy(data.start_location_x, data.start_location_y))) / 1000.0

        data['time_offset_acquisition'] = abs(data.t0 - data.t1)
        data['ice_conc'] = data['ice_conc'] * 100.0
        data[(data['freeboard'] > 2.0) | (data['freeboard'] < -0.2)] = np.nan

        var = ['freeboard', 'dist_acquisition', 'time_offset_acquisition', 'ice_conc']
        var_rename = ['laser_freeboard', 'dist_acquisition', 'time_offset_acquisition', 'sea_ice_concentration']
        master = gridding_lib.grid_data(data, grid, var, var_rename, fill_nan=True)
        master.drop(columns=['dist_acquisition_std', 'time_offset_acquisition_std', 'sea_ice_concentration_std'],
                    inplace=True)

        centroidseries = master['geometry'].centroid
        master['yc'], master['xc'] = [list(t) for t in zip(*map(stacking_lib.getXY, centroidseries))]
        master['longitude'], master['latitude'] = stacking_lib.reproject(master['yc'], master['xc'], 'epsg:3413', 'epsg:4326')
        master = master.set_index(['xc', 'yc'])
        master.drop(columns=['geometry'], inplace=True)
        master = xr.Dataset.from_dataframe(master)
        master = master.set_coords(("longitude", "latitude"))

        master = gridding_lib.set_attrbs(master, config["options"]["sensor"])

        outfile = (re.split('-', os.path.basename(file))[0] + "-" +
                   re.split('-', os.path.basename(file))[1] + "-" +
                   re.split('-', os.path.basename(file))[2] + "-" +
                   re.split('-', os.path.basename(file))[3] + "-" +
                   "{:.0f}".format(grid.geometry[0].length / 4 / 100.0) + '.nc')

        master.to_netcdf(path=out_dir + outfile)



