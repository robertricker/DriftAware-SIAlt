import stacking_lib
import pandas as pd
import geopandas as gpd
import shapely as shp
import numpy as np
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import griddata
from read_icesat2.read_ICESat2_ATL10 import read_HDF5_ATL10
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

warnings.filterwarnings('ignore')
logging.disable(sys.maxsize)
warnings.filterwarnings("ignore")


def define_grid(xmin, ymin, xmax, ymax, n_cells, epsg):
    cell_size = (xmax - xmin) / n_cells
    print('grid cell size: ' + str(cell_size) + ' m')
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax, cell_size):
        for y0 in np.arange(ymin, ymax, cell_size):
            # bounds
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            grid_cells.append(shp.geometry.box(x0, y0, x1, y1))
    return gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=epsg), cell_size


def grid_data(gdf,grid,var,var_str,fill_nan=False):
    tmp_grid = grid.copy()
    merged = gpd.sjoin(gdf[var + ['geometry']].copy(), grid, how='left', op='within')
    dissolve_mean = merged.dissolve(by='index_right', aggfunc=np.mean)
    dissolve_std = merged.dissolve(by='index_right', aggfunc=np.std)
    for i in range(0,len(var)):
        tmp_grid.loc[dissolve_mean.index, var_str[i]] = dissolve_mean[var[i]].values
        tmp_grid.loc[dissolve_std.index, var_str[i]+'_std'] = dissolve_std[var[i]].values
    if not fill_nan:
        tmp_grid = tmp_grid.dropna()
    centroidseries = tmp_grid['geometry'].centroid
    tmp_grid['x'],tmp_grid['y'] = [list(t) for t in zip(*map(stacking_lib.getXY, centroidseries))]
    tmp_grid = tmp_grid.set_index(['x', 'y'])
    return tmp_grid


def set_attrbs(xarray, sensor):
    if sensor == 'cs2':
        xarray.sea_ice_freeboard.attrs = {
            'long_name': 'sea ice elevation above sea level', 'units': 'm'}
        xarray.sea_ice_freeboard.attrs = {
            'long_name': 'standard deviation of binned sea ice freeboard estimates', 'units': 'm'}
    else:
        xarray.laser_freeboard.attrs = {
            'long_name': 'sea ice elevation above sea level', 'units': 'm'}
        xarray.laser_freeboard_std.attrs = {
            'long_name': 'standard deviation of binned sea ice freeboard estimates', 'units': 'm'}

    xarray.xc.attrs = {'long_name': 'x coordinate of projection (eastings)',
                       'units': 'm'}

    xarray.yc.attrs = {'long_name': 'y coordinate of projection (northings)',
                       'units': 'm'}

    xarray.longitude.attrs = {'long_name': 'longitude coordinate',
                              'units': 'degrees_east'}

    xarray.latitude.attrs = {'long_name': 'latitude coordinate',
                             'units': 'degrees_north'}

    xarray.laser_freeboard.attrs = {'long_name': 'sea ice elevation above sea level',
                                    'units': 'm'}

    xarray.laser_freeboard_std.attrs = {'long_name': 'standard deviation of binned sea ice freeboard estimates',
                                        'units': 'm'}

    xarray.dist_acquisition.attrs = {'long_name': 'distance from the location of data aquisition',
                                     'units': 'km'}

    xarray.time_offset_acquisition.attrs = {'long_name': 'time offfset to data aquisition',
                                            'units': 'days'}

    xarray.sea_ice_concentration.attrs = {'long_name': 'sea ice concentraion',
                                          'units': 'percentage'}
    return xarray
