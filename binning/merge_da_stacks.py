import datetime
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import MultiPoint
import numpy as np
import pandas as pd
import glob
import os
import re
import sys
import multiprocessing as mp
import json
from loguru import logger
from shapely.geometry import MultiPoint
from scipy.spatial import cKDTree
from gridding import gridding_lib

from io_tools import create_out_dir
from io_tools import init_logger
from io_tools import read_dasit_csv
from io_tools import make_csv_filename
from binning.drift_aware_uncertainties import get_neighbor_dyn_range

def extract_date_from_filename(filename):
    """Get the date YYYY-MM-DD from a csv file"""
    match = re.search(r"\d{4}\d{2}\d{2}", filename)
    return match.group(0) if match else None


def get_matching_file(bin_csv_file, search_dir):
    """Get the matching file in the search directory based on the bin_csv_file name"""
    target_date = extract_date_from_filename(os.path.basename(bin_csv_file))    
    if not target_date:
        logger.error(f"No date found in filename: {bin_csv_file}")
        return None
    for fname in os.listdir(search_dir):
        fpath = os.path.join(search_dir, fname)
        if os.path.isfile(fpath) and extract_date_from_filename(fname) == target_date:
            return fpath

def load_geodata(csv_path):
    """Load a CSV file into a GeoDataFrame with geometry from WKT format."""
    df = pd.read_csv(csv_path, skiprows=1)
    epsg_code = int(df.columns[0].split(":")[1].strip())
    df['geometry'] = df['geometry'].apply(loads)
    return gpd.GeoDataFrame(df, geometry='geometry')

def add_first_xy_columns(gdf):
    """Add 'x_first' and 'y_first' columns to the GeoDataFrame based on the first point of MultiPoint geometries."""
    gdf = gdf.copy()
    gdf['x_first'] = gdf['geometry'].apply(
        lambda geom: geom.geoms[0].x if isinstance(geom, MultiPoint) else None
    )
    gdf['y_first'] = gdf['geometry'].apply(
        lambda geom: geom.geoms[0].y if isinstance(geom, MultiPoint) else None
    )
    return gdf


def merge_on_first_point(gdf1, gdf2, columns_to_add):
    """Merge two GeoDataFrames on the first point of their geometries."""
    gdf1 = add_first_xy_columns(gdf1)
    gdf2 = add_first_xy_columns(gdf2)
    merged = pd.merge(
        gdf1,
        gdf2[columns_to_add + ['x_first', 'y_first', 'geometry', 't0']],
        how='inner',
        on=['x_first', 'y_first', 't0']
    )
    merged_gdf = gpd.GeoDataFrame(merged, geometry='geometry_y')
    merged_gdf = merged_gdf.rename(columns={'geometry_y': 'geometry'})
    return merged_gdf


def merge_bin_da_csv_files(config, gdf_bin, bin_csv):
    dir_stack_csv = os.path.dirname(os.path.dirname(config['output_dir']['trajectories'])) + '/' + config['options']['proc_step_options']['binning']['csv_dir']
    file_da_path = get_matching_file(bin_csv, dir_stack_csv)

    if file_da_path is None:
        print("No matching file found in the directory.")
        return

    print(f"Matching file founded : {file_da_path}")

    gdf_da = read_dasit_csv(file_da_path)

    print("Merging GeoDataFrames on the first point of their geometries...")
    target_var = config["options"]["target_variable"]
    columns_to_add =  ['drift_unc']

    merged = merge_on_first_point(gdf_bin, gdf_da, columns_to_add)
    # Remove the origin geometry column 
    merged = merged.drop(columns=['geometry_x'])
    merged = gpd.GeoDataFrame(merged, geometry='geometry')
    # Reorder the columns to have 'geometry' first
    merged = merged[['geometry'] + [col for col in merged.columns if col != 'geometry']]
    print(f"Merged done, {len(merged)} rows in the merged GeoDataFrame.")
    print(merged.head())

    return merged