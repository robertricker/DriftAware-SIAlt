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
from data_handler.sea_ice_thickness_products import SeaIceThicknessProducts
from stacking.stack_structure import StackStructure
from stacking.drift_aware_processor import DriftAwareProcessor
from stacking.drift_aware_uncertainties import get_neighbor_dyn_range
from stacking.interpolate_growth import interpolate_growth
from io_tools import create_out_dir
from io_tools import init_logger
from io_tools import read_dasit_csv
from io_tools import make_csv_filename


def merge_forward_reverse_stacks(config, grid, growth_cell_width, cell_width, list_f, list_r, j):
    init_logger(config)
    nbs = 260  # empirical estimate
    target_var = config["options"]["target_variable"]
    csv_dir = config['output_dir']['trajectories']
    out_epsg = config['options']['out_epsg']
    stk_opt = config['options']['proc_step_options']['stacking']
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

    data.crs = out_epsg
    # apply growth correction
    traj_geom = data['geometry']
    target_location = data["geometry"].apply(lambda g: g.geoms[-1])
    data["geometry"] = target_location
    if len(data["dt_days"].unique()) >= min_n_tps:
        f_growth, f_growth_unc, growth = interpolate_growth(
            data, target_var, growth_range, grid, growth_cell_width, min_n_tps, nbs)
        growth_interp = f_growth(
            np.array([np.array(data.geometry.x), np.array(data.geometry.y)]).transpose())
        growth_unc_interp = f_growth_unc(
            np.array([np.array(data.geometry.x), np.array(data.geometry.y)]).transpose())
    else:
        growth, growth_interp, growth_unc_interp = np.nan, np.nan, np.nan

    data = data.rename(columns={target_var: target_var + "_uncorrected"})
    data[target_var] = growth_interp * (-data.dt_days.to_numpy()) + data[target_var + "_uncorrected"].to_numpy()
    data[target_var + "_growth_unc"] = growth_unc_interp * abs(data.dt_days.to_numpy())
    data["growth_interpolated"] = growth_interp
    data["growth"] = growth
    data["drift_unc"] = data[target_var + '_drift_unc']
    points = np.array([data['geometry'].x, data['geometry'].y]).transpose()
    tree = cKDTree(points)
    data[target_var + '_drift_unc'] = data.apply(
        get_neighbor_dyn_range, args=(data, target_var, tree, cell_width/2), axis=1)
    data["geometry"] = traj_geom
    with open(os.path.join(csv_dir, outfile), 'w') as f:
        f.write(f"# {out_epsg}\n")
        data.to_csv(f, index=False)


def stack_proc(config, direct, grid):
    init_logger(config)
    m = 0
    dt1d = datetime.timedelta(days=1)
    # declare sensor and target variable options
    sensor = config["options"]["sensor"]
    target_var = config["options"]["target_variable"]
    add_var = config["options"]["add_variable"]
    hem = config["options"]["hemisphere"]
    out_epsg = config["options"]["out_epsg"]
    # declare stacking processing options
    stk_opt = config['options']['proc_step_options']['stacking']
    hist_n_bins = stk_opt['hist']['n_bins']
    hist_range = stk_opt['hist']['range']["freeboard" if "freeboard" in target_var else "thickness"]
    # define data structure
    stack = StackStructure(sensor, stk_opt['t_window'], stk_opt['t_length'])
    master, scheme = stack.get_master(), stack.get_scheme()

    # initialize data objects
    sit_product = SeaIceThicknessProducts(hem=hem, sensor=sensor, target_var=target_var,
                                          add_variable=add_var,
                                          out_epsg=out_epsg)
    sit_product.get_file_list(config['input_dir'][sensor])
    sit_product.get_file_dates()

    sic_product = SeaIceConcentrationProducts(hem=hem, product_id=config['options']['ice_conc_product'],
                                              out_epsg=out_epsg)
    sic_product.get_file_list(config['auxiliary']['ice_conc'][config['options']['ice_conc_product']])
    sic_product.get_file_dates()

    sid_product = SeaIceDriftProducts(hem=hem, product_id=config['options']['ice_drift_product'], out_epsg=out_epsg)
    sid_product.get_file_list(config['auxiliary']['ice_drift'][config['options']['ice_drift_product']])
    sid_product.get_file_dates()

    if direct == 'f':
        d_sgn = 1
        d_sgn_drift = 1
        day_range = range(0, stk_opt['t_length'], d_sgn)
    else:
        d_sgn = -1
        d_sgn_drift = 0
        day_range = range(stk_opt['t_length'] - 1, -1, d_sgn)

    # initialize drift aware processor
    processor = DriftAwareProcessor(sit_product, master=master, scheme=scheme, grid=grid)

    for i in day_range:
        processor.i = i
        t0 = stk_opt['t_start'] + datetime.timedelta(days=i)
        t1 = stk_opt['t_start'] + datetime.timedelta(days=i + 1)
        sit_product.get_target_files(t0, t1)
        sic_product.target_files = sic_product.get_target_files(t0, t1)
        if sit_product.target_files and sic_product.target_files:
            logger.info(t0.strftime("%Y%m%d") + ': altimetry files (n): ' + str(len(sit_product.target_files)))
            logger.info(t0.strftime("%Y%m%d") + ': ice_conc file day0: ' + os.path.basename(sic_product.target_files))
            sit_product.get_product()
            sic_product.ice_conc = sic_product.get_ice_concentration(sic_product.target_files)
            processor.baseline_proc(sic_product, hist_n_bins, hist_range)

        sic_product.target_files = sic_product.get_target_files(t0 + d_sgn * dt1d, t1 + d_sgn * dt1d)
        sid_product.target_files = sid_product.get_target_files(t0 + d_sgn_drift * dt1d, t1 + d_sgn_drift * dt1d)

        if sic_product.target_files and sid_product.target_files:
            logger.info(t0.strftime("%Y%m%d") + ': ice_conc file day'+str(d_sgn)+': ' +
                        os.path.basename(sic_product.target_files))
            logger.info(t0.strftime("%Y%m%d") + ': ice_drift file: ' +
                        os.path.basename(sid_product.target_files))

            sic_product.ice_conc_ahead = sic_product.get_ice_concentration(sic_product.target_files)
            sid_product.get_ice_drift(sid_product.target_files, sic_product.ice_conc_ahead)

            if (d_sgn == -1 and i > 0) or (d_sgn == 1 and i < stk_opt['t_length'] - 1):
                m = processor.drift_aware_proc(sid_product, sic_product, stk_opt['t_window'], d_sgn, day_range[0])

        gdf_final = processor.concat_gdfs(i, m)
        gdf_final[target_var+'_drift_unc'] = np.sqrt(gdf_final[target_var+'_drift_unc'])
        gdf_final = gdf_final.drop(columns=['xu', 'yu'])
        gdf_final["geometry"] = gdf_final["geometry"].apply(lambda gdf: MultiPoint(gdf))
        gdf_final = gpd.GeoDataFrame(gdf_final, geometry='geometry')
        gdf_final['divergence'] = gdf_final['divergence'].apply(json.dumps)
        gdf_final['shear'] = gdf_final['shear'].apply(json.dumps)

        outfile = make_csv_filename(config, t0, direct)
        logger.info(t0.strftime("%Y%m%d")+': generated csv file: ' + outfile)
        gdf_final['divergence'] = gdf_final['divergence'].apply(
            lambda s: s.replace('[', '').replace(']', '').replace(',', ''))
        gdf_final['shear'] = gdf_final['shear'].apply(
            lambda s: s.replace('[', '').replace(']', '').replace(',', ''))

        # optional for Luisa, save only last file
        # if abs(gdf_final['dt_days']).max()+1 == config['options']['proc_step_options']['stacking']['t_window']:
        with open(os.path.join(config['output_dir']['trajectories'], outfile), 'w') as f:
            f.write(f"# {out_epsg}\n")
            gdf_final.to_csv(f, index=False)

    return scheme


def stacking(config):
    sensor = config["options"]["sensor"]
    target_var = config["options"]["target_variable"]
    stk_opt = config['options']['proc_step_options']['stacking']
    multiproc = stk_opt['multiproc']
    parcel_grid_opt = stk_opt['parcel_grid']
    growth_grid_opt = stk_opt['growth_estimation']['growth_grid']
    csv_dir = config['output_dir']['trajectories']
    grid, cell_width = gridding_lib.define_grid(parcel_grid_opt["bounds"],
                                                parcel_grid_opt["dim"],
                                                config['options']['out_epsg'],
                                                grid_type='circular')
    growth_grid, growth_cell_width = gridding_lib.define_grid(growth_grid_opt["bounds"],
                                                              growth_grid_opt["dim"],
                                                              config['options']['out_epsg'])

    logger.info('grid cell width of the stacking parcel grid: ' + str(cell_width) + ' m')

    if stk_opt['t_length'] == 'all':
        years = list(filter(lambda f: not f.startswith('.'), os.listdir(config['dir'][sensor]['level2'][target_var])))
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
                pool.apply_async(stack_proc, args=(config, mode, grid))
            pool.close()
            pool.join()
        else:
            for mode in stk_opt['mode']:
                stack_proc(config, mode, grid)

        logger.info('start merging forward and reverse stacks')
        list_f = sorted(glob.glob(os.path.join(config['output_dir']['trajectories'], f'*_F-*.csv')))
        list_r = sorted(glob.glob(os.path.join(config['output_dir']['trajectories'], f'*_R-*.csv')))
        if multiproc:
            pool = mp.Pool(stk_opt['num_cpus'])
            for j in range(stk_opt['t_length']):
                pool.apply_async(
                    merge_forward_reverse_stacks, args=(
                        config, growth_grid, growth_cell_width, cell_width, list_f, list_r, j))
            pool.close()
            pool.join()
        else:
            for j in range(stk_opt['t_length']):
                merge_forward_reverse_stacks(config, growth_grid, growth_cell_width, cell_width, list_f, list_r, j)
