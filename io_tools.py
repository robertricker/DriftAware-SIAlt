import pyproj
import netCDF4
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import os
import re
from shapely.geometry import Point
from shapely.wkt import loads
from typing import Tuple
from scipy.interpolate import griddata
from datetime import datetime
from loguru import logger
import sys


TRAJECTORY_FORMAT_VERSION = 1


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


def get_sea_ice_regions(file, netcdf_bounds, cell_width, grid_epsg, hemisphere):
    xmin, xmax = netcdf_bounds[0], netcdf_bounds[2]
    ymin, ymax = netcdf_bounds[1], netcdf_bounds[3]

    x_range = np.arange(np.floor(xmin), np.ceil(xmax), cell_width) + cell_width / 2
    y_range = np.arange(np.floor(ymin), np.ceil(ymax), cell_width) + cell_width / 2
    xc, yc = np.meshgrid(x_range, y_range)
    lon_grid, lat_grid = transform_coords(np.ma.getdata(xc),
                                          np.ma.getdata(yc),
                                          grid_epsg, 'epsg:4326')

    dict_region = {'epsg': {'sh': 'epsg:6932',
                'nh': 'epsg:6931'}, 'var' : {'nh' : 'sea_ice_region', 'sh': 'sea_ice_region_NASA_modified'}}

    with netCDF4.Dataset(file) as reg_data:
        xc, yc = np.meshgrid(np.ma.getdata(reg_data.variables['x'][:]),
                             np.ma.getdata(reg_data.variables['y'][:]))
        region_variable = reg_data.variables[dict_region['var'][hemisphere]]
        value = np.ma.getdata(region_variable[:, :]).flatten()
        region_metadata = {
            'long_name': region_variable.getncattr('long_name'),
            'flag_values': np.asarray(region_variable.getncattr('flag_values')),
            'flag_meanings': region_variable.getncattr('flag_meanings'),
            'source': os.path.splitext(os.path.basename(file))[0],
        }

    lon, lat = transform_coords(np.ma.getdata(xc).flatten(),
                                np.ma.getdata(yc).flatten(),
                                dict_region['epsg'][hemisphere], 'epsg:4326')
    coords = np.transpose(np.vstack((lon, lat)))
    region = griddata(coords, value, (lon_grid, lat_grid), method='nearest')
    return region, region_metadata


def create_out_dir(config, parent_directory, cell_width, source_stack=None):
    target_variable = config["options"]["target_variable"]
    hem = config["options"]["hemisphere"]
    stage = config['stage']
    if stage == 'gridding':
        if source_stack is None:
            source_stack = config['gridding'].get('source_stack')
        if source_stack is None:
            raise ValueError('trajectory source metadata is required for gridding')
        t_window = source_stack['window_days']
        mode = source_stack['mode']
    else:
        t_window = config['stacking']['t_window']
        mode = config['stacking']['mode']
    epsg = 'epsg' + config['options']['out_epsg'].split(":")[1]
    res = "{:.0f}".format(cell_width / 100.0)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if stage == 'gridding':
        dt_days_max = config['gridding']['dt_days_max']
        sub_dir_name = f'{target_variable}-{hem}-{t_window}{mode}-{epsg}_{res}_{dt_days_max}-{timestamp}'
    else:
        sub_dir_name = f'{target_variable}-{hem}-{t_window}{mode}-{epsg}_{res}-{timestamp}'
    sub_dir_path = os.path.join(parent_directory, sub_dir_name)
    os.makedirs(sub_dir_path)
    return sub_dir_path + '/'


def init_logger(config):
    logger.remove()
    logger.add(sys.stdout, colorize=True,
               format=("<green>{time:YYYY-MM-DDTHH:mm:ss}</green> "
                       "<blue>{module}</blue> "
                       "<cyan>{function}</cyan> {message}"),
               enqueue=True)
    logger.add(config['logging'],
               format="{time:YYYY-MM-DDTHH:mm:ss} {module} {function} {message}", enqueue=True)


def build_trajectory_metadata(config):
    """Build the metadata stored in the first line of a trajectory CSV."""
    stack_options = config['stacking']

    return {
        'format_version': TRAJECTORY_FORMAT_VERSION,
        'crs': config['options']['out_epsg'],
        'target_variable': config['options']['target_variable'],
        'source_stack': {
            'mode': stack_options['mode'],
            'window_days': stack_options['t_window'],
            'histogram': {
                'n_bins': stack_options['hist']['n_bins'],
                'range': stack_options['hist']['range'],
            },
        },
    }


def write_dasit_csv(data, file, config):
    """Write a trajectory CSV with a versioned JSON metadata header."""
    metadata = build_trajectory_metadata(config)
    with open(file, 'w') as stream:
        stream.write('# ' + json.dumps(metadata, separators=(',', ':')) + '\n')
        data.to_csv(stream, index=False)


def read_dasit_metadata(file):
    """Read trajectory metadata, including the legacy CRS-only header."""
    with open(file, 'r') as stream:
        first_line = stream.readline().strip()

    if not first_line.startswith('#'):
        raise ValueError(f'trajectory CSV has no metadata header: {file}')

    payload = first_line[1:].strip()
    if payload.startswith('{'):
        metadata = json.loads(payload)
        version = metadata.get('format_version')
        if version != TRAJECTORY_FORMAT_VERSION:
            raise ValueError(
                f'unsupported trajectory format_version {version!r} in {file}')
        if not metadata.get('crs'):
            raise ValueError(f'trajectory metadata has no CRS: {file}')
        return metadata

    # Backward compatibility with headers such as ``# EPSG:6931``.
    match = re.fullmatch(r'(?:EPSG\s*:\s*)?(\d+)', payload, re.IGNORECASE)
    if not match:
        raise ValueError(f'unrecognized trajectory metadata header in {file}')
    return {
        'format_version': 0,
        'crs': f'EPSG:{match.group(1)}',
    }


def read_dasit_csv(file, return_metadata=False):
    metadata = read_dasit_metadata(file)
    data = pd.read_csv(file, skiprows=1)
    data['geometry'] = data['geometry'].apply(loads)
    data = gpd.GeoDataFrame(data, geometry='geometry')
    data.set_crs(metadata['crs'], allow_override=True, inplace=True)
    if return_metadata:
        return data, metadata
    return data


def make_csv_filename(config, t0, direct):
    prefix = config['stacking']['filename_prefix']
    prdlvl = 'L2P'
    var_map = {
        "sea_ice_thickness": "SITHICK",
        "sea_ice_freeboard": "SIFB",
        "radar_freeboard": "RFB",
        "total_freeboard": "TFB"}
    var = var_map.get(config['options']['target_variable'])
    instr_map = {
        "envisat": "RA2_ENVISAT",
        "cryosat2": "SIRAL_CRYOSAT2",
        "sentinel3a": "SRAL_SENTINEL3A",
        "sentinel3b": "SRAL_SENTINEL3B",
        "icesat2": "ATLAS_ICESAT2"}
    
    instr_list = [instr_map.get(config['options']['sensor'][i]) for i in range(len(config['options']['sensor']))]
    instr = "_".join(instr_list)
    region = config['options']['hemisphere'].upper()
    mode = 'DA_'+direct.upper()
    period = t0.strftime('%Y%m%d')
    version = f"fv{config['version']}"

    return f"{prefix}-{prdlvl}-{mode}-{var}-{instr}-{region}-{period}-{version}.csv"
