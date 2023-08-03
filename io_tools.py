import pyproj
import netCDF4
import numpy as np
import os
from shapely.geometry import Point
from typing import Tuple
from scipy.interpolate import griddata
from datetime import datetime


def transform_coords(x: float, y: float, in_epsg: str, out_epsg: str) -> Tuple[float, float]:
    """
    Transforms coordinates from one projection to another using pyproj.

    Args:
        x: The x-coordinate.
        y: The y-coordinate.
        in_epsg: The EPSG code of the input projection.
        out_epsg: The EPSG code of the output projection.

    Returns:
        A tuple of the transformed (x, y) coordinates.
    """
    source_crs = pyproj.CRS(in_epsg)
    destination_crs = pyproj.CRS(out_epsg)
    transformer = pyproj.Transformer.from_crs(source_crs, destination_crs, always_xy=True)
    return transformer.transform(x, y)


def get_sea_ice_regions(file, netcdf_bounds, cell_width, grid_epsg):
    xmin, xmax = netcdf_bounds[0], netcdf_bounds[2]
    ymin, ymax = netcdf_bounds[1], netcdf_bounds[3]

    x_range = np.arange(np.floor(xmin), np.ceil(xmax), cell_width) + cell_width / 2
    y_range = np.arange(np.floor(ymin), np.ceil(ymax), cell_width) + cell_width / 2
    xc, yc = np.meshgrid(x_range, y_range)
    lon_grid, lat_grid = transform_coords(np.ma.getdata(xc),
                                          np.ma.getdata(yc),
                                          grid_epsg, 'epsg:4326')

    reg_data = netCDF4.Dataset(file)
    xc, yc = np.meshgrid(np.ma.getdata(reg_data.variables['x'][:]),
                         np.ma.getdata(reg_data.variables['y'][:]))
    lon, lat = transform_coords(np.ma.getdata(xc).flatten(),
                                np.ma.getdata(yc).flatten(),
                                'epsg:6931', 'epsg:4326')
    value = np.ma.getdata(reg_data.variables['sea_ice_region'][:, :]).flatten()
    coords = np.transpose(np.vstack((lon, lat)))
    region = griddata(coords, value, (lon_grid, lat_grid), method='nearest')
    return region


def create_subdir(config, parent_directory, grid):
    target_variable = config["options"]["target_variable"]
    hem = config["options"]["hemisphere"]
    stk_opt = config['options']['proc_step_options']['stacking']
    t_window = stk_opt['t_window']
    mode = stk_opt['mode']
    epsg = 'epsg' + config['options']['out_epsg'].split(":")[1]
    res = "{:.0f}".format(grid.geometry[0].length / 4 / 100.0)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    sub_dir_name = f'{target_variable}-{hem}-{t_window}{mode}-{epsg}_{res}-{timestamp}'
    sub_dir_path = os.path.join(parent_directory, sub_dir_name)
    os.makedirs(sub_dir_path)
    return sub_dir_path


def create_out_dir(config, parent_directory, grid):
    target_variable = config["options"]["target_variable"]
    hem = config["options"]["hemisphere"]
    stk_opt = config['options']['proc_step_options']['stacking']
    t_window = stk_opt['t_window']
    mode = stk_opt['mode']
    epsg = 'epsg' + config['options']['out_epsg'].split(":")[1]
    res = "{:.0f}".format(grid.geometry[0].length / 4 / 100.0)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    sub_dir_name = f'{target_variable}-{hem}-{t_window}{mode}-{epsg}_{res}-{timestamp}'
    sub_dir_path = os.path.join(parent_directory, sub_dir_name)
    os.makedirs(sub_dir_path)
    return sub_dir_path + '/'

