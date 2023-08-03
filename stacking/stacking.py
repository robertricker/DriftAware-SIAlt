import datetime
import geopandas as gpd
import numpy as np
import pandas as pd
import glob
import os
import re
import sys
import multiprocessing as mp
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


def merge_forward_reverse_stacks(config, grid, cell_width):
    nbs = 260
    target_var = config["options"]["target_variable"]
    geojson_dir = config["dir"][config["options"]["sensor"]]["geojson"]
    out_epsg = config['options']['out_epsg']
    stk_opt = config['options']['proc_step_options']['stacking']
    start_date = stk_opt['t_start']
    t_series_length = stk_opt['t_length']

    growth_range = stk_opt['growth_estimation']['growth_range']["freeboard" if "free" in target_var else "thickness"]
    min_n_tps = stk_opt['growth_estimation']['min_n_tiepoints']

    list_f = sorted(glob.glob(os.path.join(geojson_dir, f'*-f-*.geojson')))
    list_r = sorted(glob.glob(os.path.join(geojson_dir, f'*-r-*.geojson')))

    for j in range(t_series_length):
        cday = start_date + datetime.timedelta(days=j)
        subs = cday.strftime("%Y%m%d")
        logger.info("finalizing geojson file on day: " + subs)
        if stk_opt['mode'] == 'fr':
            file_f = [i for i in list_f if subs in re.search('-(.+?)-*.geojson', os.path.basename(i)).group(1)]
            file_r = [i for i in list_r if subs in re.search('-(.+?)-*.geojson', os.path.basename(i)).group(1)]
            stack_f = gpd.read_file(file_f[0])
            stack_r = gpd.read_file(file_r[0])
            stack_r = stack_r[stack_r.dt_days != 0].reset_index(drop=True)
            data = pd.concat([stack_f, stack_r], ignore_index=True)
            out_file = os.path.basename(file_f[0]).replace("-f-", "-")
            os.remove(file_f[0])
            os.remove(file_r[0])
        else:
            listfr = list_f + list_r
            file = [i for i in listfr if subs in re.search('-(.+?)-*.geojson', os.path.basename(i)).group(1)]
            data = gpd.read_file(file[0])
            out_file = os.path.basename(file[0])
            os.remove(file[0])

        data.crs = out_epsg
        # apply growth correction
        traj_geom = data['geometry']
        target_location = data["geometry"].apply(lambda g: g.geoms[-1])
        data["geometry"] = target_location
        f_growth, f_growth_unc, _ = interpolate_growth(data, target_var, growth_range, grid, cell_width, min_n_tps, nbs)
        growth_interp = f_growth(np.array([np.array(data.geometry.x), np.array(data.geometry.y)]).transpose())
        growth_unc_interp = f_growth_unc(np.array([np.array(data.geometry.x), np.array(data.geometry.y)]).transpose())
        data[target_var + "_corr"] = growth_interp * (-data.dt_days.to_numpy()) + data[target_var].to_numpy()
        data[target_var + "_growth_unc"] = (growth_unc_interp/np.sqrt(nbs)) * abs(data.dt_days.to_numpy())
        data["growth_interpolated"] = growth_interp

        points = np.array([data['geometry'].x, data['geometry'].y]).transpose()
        tree = cKDTree(points)

        data["sea_ice_thickness_drift_unc"] = data.apply(get_neighbor_dyn_range, args=(data, target_var, tree), axis=1)
        data["geometry"] = traj_geom
        data.to_file(os.path.join(geojson_dir, out_file), driver="GeoJSON")


def stack_proc(config, direct, grid):
    m = 0
    dt1d = datetime.timedelta(days=1)

    logger.remove()
    logger.add(sys.stdout, colorize=True,
               format=("<green>{time:YYYY-MM-DDTHH:mm:ss}</green> "
                       "<blue>{module}</blue> "
                       "<cyan>{function}</cyan> {message}"),
               enqueue=True)
    logger.add(config['dir']['logging'],
               format="{time:YYYY-MM-DDTHH:mm:ss} {module} {function} {message}", enqueue=True)

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
    sit_product.get_file_list(config['dir'][sensor]['level2'][target_var])
    sit_product.get_file_dates()

    sic_product = SeaIceConcentrationProducts(hem=hem, product_id=config['options']['ice_conc_product'],
                                              out_epsg=out_epsg)
    sic_product.get_file_list(config['dir']['auxiliary']['ice_conc'][config['options']['ice_conc_product']])
    sic_product.get_file_dates()

    sid_product = SeaIceDriftProducts(hem=hem, product_id=config['options']['ice_drift_product'], out_epsg=out_epsg)
    sid_product.get_file_list(config['dir']['auxiliary']['ice_drift'][config['options']['ice_drift_product']])
    sid_product.get_file_dates()

    if direct == 'f':
        d_sgn = 1
        day_range = range(0, stk_opt['t_length'], d_sgn)
    else:
        d_sgn = -1
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
        sid_product.target_files = sid_product.get_target_files(t0, t1)

        if sic_product.target_files and sid_product.target_files:
            logger.info(t0.strftime("%Y%m%d") + ': ice_conc file day'+str(d_sgn)+': ' +
                        os.path.basename(sic_product.target_files))
            logger.info(t0.strftime("%Y%m%d") + ': ice_drift file: ' +
                        os.path.basename(sid_product.target_files))

            sic_product.ice_conc_ahead = sic_product.get_ice_concentration(sic_product.target_files)
            sid_product.get_ice_drift(sid_product.target_files, sic_product.ice_conc)

            if (d_sgn == -1 and i > 0) or (d_sgn == 1 and i < stk_opt['t_length'] - 1):
                m = processor.drift_aware_proc(sid_product, sic_product, stk_opt['t_window'], d_sgn, day_range[0])

        gdf_final = processor.concat_gdfs(i, m)
        gdf_final[target_var+'_drift_unc'] = np.sqrt(gdf_final[target_var+'_drift_unc'])
        gdf_final = gdf_final.drop(columns=['xu', 'yu'])
        gdf_final["geometry"] = gdf_final["geometry"].apply(lambda gdf: MultiPoint(gdf))

        outfile = (
            f"{target_var}-{sensor}-{hem}-{t0.strftime('%Y%m%d')}-"
            f"{config['version']}-{direct}-epsg{out_epsg.split(':')[1]}_"
            f"{grid.geometry[0].length / 4 / 100.0:.0f}.geojson")

        logger.info(t0.strftime("%Y%m%d")+': generated geojson file: ' + outfile)
        gdf_final.to_file(config['dir'][sensor]['geojson'] + outfile, driver="GeoJSON")

    return scheme


def stacking(config):
    sensor = config["options"]["sensor"]
    stk_opt = config['options']['proc_step_options']['stacking']
    multiproc = stk_opt['multiproc']
    parcel_grid_opt = stk_opt['parcel_grid']
    growth_grid_opt = stk_opt['growth_estimation']['growth_grid']

    grid, cell_width = gridding_lib.define_grid(parcel_grid_opt["bounds"],
                                                parcel_grid_opt["dim"],
                                                config['options']['out_epsg'])

    config['dir'][sensor]['geojson'] = create_out_dir(config, config['dir'][sensor]['geojson'], grid)
    logger.info('grid cell size of the stacking parcel grid: ' + str(cell_width) + ' m')

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
    grid, cell_width = gridding_lib.define_grid(growth_grid_opt["bounds"],
                                                growth_grid_opt["dim"],
                                                config['options']['out_epsg'])

    merge_forward_reverse_stacks(config, grid, cell_width)
