import datetime
import geopandas as gpd
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
from data_handler.sea_ice_concentration_products import SeaIceConcentrationProducts
from data_handler.sea_ice_drift_products import SeaIceDriftProducts
from data_handler.sea_ice_thickness_products import SeaIceThicknessMultiProducts
from data_handler.sea_ice_thickness_products import SeaIceThicknessProducts

from data_handler.air_temperature_products import AirTemperatureProducts
from data_handler.ocean_heat_flux_products import OceanHeatFluxProducts
from stacking.stack_structure import StackStructure
from binning.binning_structure import BinningStructure

from binning.binning_processor import BinningProcessor
from binning.drift_aware_uncertainties import get_neighbor_dyn_range_postproc
from stacking.interpolate_growth import interpolate_growth
from io_tools import create_out_dir
from io_tools import init_logger
from io_tools import read_dasit_csv
from io_tools import make_csv_filename
from binning.merge_da_stacks import merge_bin_da_csv_files

def merge_forward_reverse_da_stacks_compute_unc(config, grid, growth_cell_width, cell_width, list_f, list_r, j):
    init_logger(config)
    nbs = 260  # empirical estimate
    target_var = config["options"]["target_variable"]
    csv_dir = config['output_dir']['trajectories']
    out_epsg = config['options']['out_epsg']
    stk_opt = config['options']['proc_step_options']['binning']
    start_date = stk_opt['t_start']
    growth_range = stk_opt['growth_estimation']['growth_range']["freeboard" if "free" in target_var else "thickness"]
    min_n_tps = stk_opt['growth_estimation']['min_n_tiepoints']
    cday = start_date + datetime.timedelta(days=j)
    subs = cday.strftime("%Y%m%d")
    logger.info("finalizing csv file on day: " + subs)
    if stk_opt['mode'] == 'fr':
        file_f = [i for i in list_f if subs in re.search('-(.+?)-*.csv', os.path.basename(i)).group(1)]
        file_r = [i for i in list_r if subs in re.search('-(.+?)-*.csv', os.path.basename(i)).group(1)]
        stack_f = read_dasit_csv(file_f[0])
        stack_r = read_dasit_csv(file_r[0])
        stack_r = stack_r[stack_r.dt_days != 0].reset_index(drop=True)
        data = pd.concat([stack_f, stack_r], ignore_index=True)
        outfile = os.path.basename(file_f[0]).replace("_F-", "-")
        os.remove(file_f[0])
        os.remove(file_r[0])
    else:
        listfr = list_f + list_r
        file = [i for i in listfr if subs in re.search('-(.+?)-*.csv', os.path.basename(i)).group(1)]
        data = read_dasit_csv(file[0])
        outfile = os.path.basename(file[0])
        os.remove(file[0])
    outfile_density = 'density_' + outfile
    data.crs = out_epsg
    # apply growth correction
    
    
    #data[target_var + '_drift_unc'] = data.apply(
    #    get_neighbor_dyn_range, args=(data, target_var, tree, cell_width/2), axis=1)


    merged = merge_bin_da_csv_files(config, data, outfile)

    traj_geom = merged['geometry']
    target_location = merged["geometry"].apply(lambda g: g.geoms[-1])
    merged["geometry"] = target_location

    if target_var == 'sea_ice_thickness':
        if len(merged["dt_days"].unique()) >= min_n_tps:
            f_growth, f_growth_unc, growth, nb_tie_points, counts = interpolate_growth(
                merged, target_var, growth_range, grid, growth_cell_width, min_n_tps, nbs, config["options"]["hemisphere"])
            growth_interp = f_growth(
                np.array([np.array(merged.geometry.x), np.array(merged.geometry.y)]).transpose())
            growth_unc_interp = f_growth_unc(
                np.array([np.array(merged.geometry.x), np.array(merged.geometry.y)]).transpose())
        else:
            growth, growth_interp, growth_unc_interp, counts = np.nan, np.nan, np.nan, np.nan


        merged[target_var] = growth_interp * (-merged.dt_days.to_numpy()) + merged[target_var + "_uncorrected"].to_numpy()
        merged[target_var + "_growth_unc"] = growth_unc_interp * abs(merged.dt_days.to_numpy())
        merged["growth_interpolated"] = growth_interp
        merged["growth"] = growth

    merged = merged.rename(columns={target_var: target_var + "_uncorrected"})

    points = np.array([merged['geometry'].x, merged['geometry'].y]).transpose()
    tree = cKDTree(points)
    if target_var == 'sea_ice_thickness':
        merged[target_var + '_drift_unc'] = merged.apply(get_neighbor_dyn_range_postproc, args=(merged, target_var, tree, cell_width/2), axis=1)
    else:
        merged[target_var + '_drift_unc'] = merged.apply(get_neighbor_dyn_range_postproc, args=(merged, target_var + "_uncorrected", tree, cell_width/2), axis=1)
    merged["geometry"] = traj_geom
    if target_var == 'sea_ice_thickness':
        merged[target_var + '_total_unc'] = np.sqrt(merged[target_var+'_drift_unc']**2 +
                                                    merged[target_var+'_l2_unc']**2+
                                                    merged[target_var+'_growth_unc']**2)
    else:
        merged[target_var + '_total_unc'] = np.sqrt(merged[target_var+'_drift_unc']**2 +
                                                    merged[target_var+'_l2_unc']**2)
        
    with open(os.path.join(csv_dir, outfile), 'w') as f:
        f.write(f"# {out_epsg}\n")
        merged.to_csv(f, index=False)
    
    # Only if you want to save the density of point per lat band
    # if type(counts)!=float:
    #     with open(os.path.join(csv_dir, outfile_density), 'w') as f:
    #         f.write(f"# {out_epsg}\n")
    #         counts.to_csv(f, index=False)


def binning_proc(config, direct, grid):
    init_logger(config)
    m = 0
    dt1d = datetime.timedelta(days=1)
    # declare sensor and target variable options
    sensor = config["options"]["sensor"]
    target_var = config["options"]["target_variable"]
    add_var = config["options"]["add_variable"]
    hem = config["options"]["hemisphere"]
    out_epsg = config["options"]["out_epsg"]
    # declare binning processing options
    stk_opt = config['options']['proc_step_options']['binning']
    hist_n_bins = stk_opt['hist']['n_bins']
    hist_range = stk_opt['hist']['range']["freeboard" if "freeboard" in target_var else "thickness"]
    # define data structure
    stack = StackStructure(sensor, stk_opt['t_window'], stk_opt['t_length'])
    master, scheme = stack.get_master(), stack.get_scheme()

    # initialize data objects
    sit_product = SeaIceThicknessMultiProducts(hem=hem, sensor=sensor, target_var=target_var,
                                          add_variable=add_var,
                                          out_epsg=out_epsg)
    sit_product.get_file_list(config['input_dir'])
    sit_product.get_file_dates()

    if direct == 'f':
        d_sgn = 1
        day_range = range(0, stk_opt['t_length'], d_sgn)
    else:
        d_sgn = -1
        day_range = range(stk_opt['t_length'] - 1, -1, d_sgn)

    # initialize drift aware processor
    processor = BinningProcessor(sit_product, master=master, scheme=scheme, grid=grid)

    # for each day we want to stack
    for i in day_range:

        processor.i = i 
        t0 = stk_opt['t_start'] + datetime.timedelta(days=i) 
        t1 = stk_opt['t_start'] + datetime.timedelta(days=i + 1) 
        sit_product.get_target_files(t0, t1) 

        # Number of empty list for missions
        empty_lists = [k for k, v in sit_product.target_files.items() if isinstance(v, list) and len(v) == 0]
        file_counts = {k: len(v) for k, v in sit_product.target_files.items() if isinstance(v, list) and len(v) > 0}
        # Build the baseline so the line that corresponds to the actual time, without any advection needed
        if len(empty_lists)==0:
            logger.info(t0.strftime("%Y%m%d") + ': altimetry files (n): ' + str(file_counts))
            sit_product.get_product()
            processor.baseline_proc( hist_n_bins, hist_range)

        if (d_sgn == -1 and i > 0) or (d_sgn == 1 and i < stk_opt['t_length'] - 1):
            m = processor.binning_proc(stk_opt['t_window'], d_sgn, day_range[0])

        gdf_final = processor.concat_gdfs(i, m)
        gdf_final = gdf_final.drop(columns=['xu', 'yu'])
        gdf_final["geometry"] = gdf_final["geometry"].apply(lambda gdf: MultiPoint(gdf))
        gdf_final = gpd.GeoDataFrame(gdf_final, geometry='geometry')
        
        outfile = make_csv_filename(config, t0, direct)
        logger.info(t0.strftime("%Y%m%d")+': generated csv file: ' + outfile)
        
        # optional for Luisa, save only last file
        # if abs(gdf_final['dt_days']).max()+1 == config['options']['proc_step_options']['stacking']['t_window']:
        with open(os.path.join(config['output_dir']['trajectories'], outfile), 'w') as f:
            f.write(f"# {out_epsg}\n")
            gdf_final.to_csv(f, index=False)

    return scheme


def binning(config):
    sensor = config["options"]["sensor"]
    stk_opt = config['options']['proc_step_options']['binning']
    multiproc = stk_opt['multiproc']
    parcel_grid_opt = stk_opt['parcel_grid']
    growth_grid_opt = stk_opt['growth_estimation']['growth_grid']
    csv_dir = config['output_dir']['trajectories']
    csv_dir = csv_dir.replace(f'{sensor}', "_".join(sensor))
    grid, cell_width = gridding_lib.define_grid(parcel_grid_opt["bounds"],
                                                parcel_grid_opt["dim"],
                                                config['options']['out_epsg'],
                                                grid_type='circular')
    growth_grid, growth_cell_width = gridding_lib.define_grid(growth_grid_opt["bounds"],
                                                              growth_grid_opt["dim"],
                                                              config['options']['out_epsg'])

    logger.info('grid cell width of the stacking parcel grid: ' + str(cell_width) + ' m')

    if stk_opt['t_length'] == 'all':
        years = list(filter(lambda f: not f.startswith('.'), os.listdir(config['input_dir'][sensor])))
        years.sort()
        years = [int(x) for x in years]
        years = [y for y in years if y >= stk_opt['t_start'].year]
    else:
        years = [stk_opt['t_start'].year]

    t_length = stk_opt['t_length']
    for yr in years:
        config['output_dir']['trajectories'] = create_out_dir(config, csv_dir, cell_width)
        stk_opt['t_start'] = stk_opt['t_start'].replace(year=yr)

        if t_length in ['season', 'all']:
            if config["options"]["hemisphere"] == 'nh':
                stk_opt['t_length'] = (
                        datetime.datetime(stk_opt['t_start'].year + 1, 5, 1, 0, 0) - stk_opt['t_start']).days
            elif config["options"]["hemisphere"] == 'sh':
                stk_opt['t_length'] = (
                        datetime.datetime(stk_opt['t_start'].year, 11, 1, 0, 0) - stk_opt['t_start']).days

        if multiproc:
            logger.info('start multiprocessing')
            pool = mp.Pool(2)
            for mode in stk_opt['mode']:
                pool.apply_async(binning_proc, args=(config, mode, grid))
            pool.close()
            pool.join()
        else:
            for mode in stk_opt['mode']:
                binning_proc(config, mode, grid)

        logger.info('start merging forward and reverse stacks')

        list_f = sorted(glob.glob(os.path.join(config['output_dir']['trajectories'], f'*_F-*.csv')))
        list_r = sorted(glob.glob(os.path.join(config['output_dir']['trajectories'], f'*_R-*.csv')))
        if multiproc:
            pool = mp.Pool(stk_opt['num_cpus'])
            for j in range(stk_opt['t_length']):
                pool.apply_async(
                    merge_forward_reverse_da_stacks_compute_unc, args=(
                        config, growth_grid, growth_cell_width, cell_width, list_f, list_r, j))
            pool.close()
            pool.join()
        else:
            for j in range(stk_opt['t_length']):
                merge_forward_reverse_da_stacks_compute_unc(config, growth_grid, growth_cell_width, cell_width, list_f, list_r, j)


def merge_with_DA(config):
    """
    Merge the binning results with the drift aware uncertainties.
    """
    init_logger(config)
    stk_opt = config['options']['proc_step_options']['binning']
    csv_dir = config['output_dir']['trajectories']
    out_epsg = config['options']['out_epsg']
    target_var = config["options"]["target_variable"]
    
    # Get the list of files
    list_files = sorted(glob.glob(os.path.join(csv_dir, f'*_{target_var}-*.csv')))
    
    for file in list_files:
        data = read_dasit_csv(file)
        data.crs = out_epsg
        
        # Apply drift aware uncertainties
        data[target_var + '_drift_unc'] = data.apply(
            get_neighbor_dyn_range, args=(data, target_var, None, stk_opt['cell_width']/2), axis=1)
        
        outfile = os.path.basename(file).replace("_F-", "-")
        with open(os.path.join(csv_dir, outfile), 'w') as f:
            f.write(f"# {out_epsg}\n")
            data.to_csv(f, index=False)
    
    logger.info('Merging with drift aware uncertainties completed.')