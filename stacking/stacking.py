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
from shapely.geometry import MultiPoint, Polygon, shape
from scipy.spatial import cKDTree
from gridding import gridding_lib
from data_handler.sea_ice_concentration_products import SeaIceConcentrationProducts
from data_handler.sea_ice_drift_products import SeaIceDriftProducts
from data_handler.sea_ice_thickness_products import SeaIceThicknessMultiProducts
from data_handler.sea_ice_thickness_products import SeaIceThicknessProducts
from data_handler.sea_ice_thickness_clim_products import SeaIceThicknessClimProducts

from data_handler.air_temperature_products import AirTemperatureProducts
from data_handler.ocean_heat_flux_products import OceanHeatFluxProducts
from stacking.stack_structure import StackStructure
from stacking.drift_aware_processor import DriftAwareProcessor
from stacking.drift_aware_uncertainties import get_neighbor_dyn_range
from stacking.interpolate_growth import interpolate_growth
from io_tools import create_out_dir
from io_tools import init_logger
from io_tools import read_dasit_csv
from io_tools import make_csv_filename
from data_handler.filter_miz import compute_apply_flag


def filter_sit_by_spatial_filter(data, bounds=None, polygon=None, vector_file=None):
    """Keep SIT observations inside one lon/lat bounding box, polygon, or vector file."""
    filters = [bounds is not None, polygon is not None, vector_file is not None]
    if sum(filters) == 0:
        return data
    if sum(filters) > 1:
        raise ValueError("Use only one of lon_lat_bounds, lon_lat_polygon, or spatial_filter_file.")
    if not {'longitude', 'latitude'}.issubset(data.columns):
        raise ValueError("The SIT product does not provide longitude and latitude columns.")

    if polygon is not None or vector_file is not None:
        if vector_file is not None:
            region = gpd.read_file(vector_file)
            if region.empty:
                raise ValueError(f"spatial_filter_file contains no geometries: {vector_file}")
            if region.crs is None:
                raise ValueError("spatial_filter_file must define its coordinate reference system.")
            region_geometry = region.to_crs("EPSG:4326").geometry.unary_union
        else:
            region_geometry = shape(polygon) if isinstance(polygon, dict) else Polygon(polygon)
        if region_geometry.is_empty or not region_geometry.is_valid:
            raise ValueError("lon_lat_polygon must define a valid polygon.")

        points = gpd.GeoSeries(
            gpd.points_from_xy(data['longitude'], data['latitude']), crs="EPSG:4326"
        )
        # covers includes points on the boundary of the region.
        mask = points.apply(region_geometry.covers)
        return data.loc[mask.to_numpy()].copy()

    # Bounding-box filter; lon_min > lon_max represents a box crossing the antimeridian.
    if len(bounds) != 4:
        raise ValueError("lon_lat_bounds must be [lon_min, lat_min, lon_max, lat_max].")

    lon_min, lat_min, lon_max, lat_max = bounds
    if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180 and
            -90 <= lat_min <= 90 and -90 <= lat_max <= 90 and lat_min <= lat_max):
        raise ValueError("lon_lat_bounds contains invalid longitude or latitude limits.")
    # Uniformise les longitudes dans [-180, 180]
    longitude = ((data["longitude"] + 180) % 360) - 180

    lat_mask = data["latitude"].between(lat_min, lat_max)

    lon_mask = (
        longitude.between(lon_min, lon_max)
        if lon_min <= lon_max
        else ((longitude >= lon_min) | (longitude <= lon_max))
    )

    return data.loc[lon_mask & lat_mask].copy()


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
        if len(file_f) == 0 or len(file_r) == 0:
            logger.warning(subs + ': No files found for this date. Skipping.')
            return
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
        if len(file) == 0:
            logger.warning(subs + ': No files found for this date. Skipping.')
            return
        data = read_dasit_csv(file[0])
        outfile = os.path.basename(file[0])
        os.remove(file[0])
    outfile_density = 'density_' + outfile
    data.crs = out_epsg
    # apply growth correction
    traj_geom = data['geometry']
    target_location = data["geometry"].apply(lambda g: g.geoms[-1])
    data["geometry"] = target_location
    if len(data["dt_days"].unique()) >= min_n_tps:
        f_growth, f_growth_unc, growth, nb_tie_points, counts = interpolate_growth(
            data, target_var, growth_range, grid, growth_cell_width, min_n_tps, nbs, config["options"]["hemisphere"])
        growth_interp = f_growth(
            np.array([np.array(data.geometry.x), np.array(data.geometry.y)]).transpose())
        growth_unc_interp = f_growth_unc(
            np.array([np.array(data.geometry.x), np.array(data.geometry.y)]).transpose())
    else:
        growth, growth_interp, growth_unc_interp, counts = np.nan, np.nan, np.nan, np.nan

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
    
    # Only if you want to save the density of point per lat band
    # if type(counts)!=float:
    #     with open(os.path.join(csv_dir, outfile_density), 'w') as f:
    #         f.write(f"# {out_epsg}\n")
    #         counts.to_csv(f, index=False)


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
    source_length = stk_opt['t_length']
    continue_without_sit = stk_opt.get('continue_tracking_without_sit', False)
    save_only_terminal_file = stk_opt.get('save_only_terminal_file', False)
    lon_lat_bounds = stk_opt.get('lon_lat_bounds')
    lon_lat_polygon = stk_opt.get('lon_lat_polygon')
    spatial_filter_file = stk_opt.get('spatial_filter_file')
    tracking_length = max(source_length, stk_opt['t_window']) if continue_without_sit else source_length
    extension_length = tracking_length - source_length
    hist_n_bins = stk_opt['hist']['n_bins']
    hist_range = stk_opt['hist']['range']["freeboard" if "freeboard" in target_var else "thickness"]
    # define data structure
    stack = StackStructure(sensor, stk_opt['t_window'], tracking_length)
    master, scheme = stack.get_master(), stack.get_scheme()

    # initialize data objects
    sit_product = SeaIceThicknessMultiProducts(hem=hem, sensor=sensor, target_var=target_var,
                                          add_variable=add_var,
                                          out_epsg=out_epsg)
    sit_product.get_file_list(config['input_dir'])
    sit_product.get_file_dates()

    sic_product = SeaIceConcentrationProducts(hem=hem, product_id=config['options']['ice_conc_product'],
                                              out_epsg=out_epsg)
    sic_product.get_file_list(config['auxiliary']['ice_conc'][config['options']['ice_conc_product']])
    sic_product.get_file_dates()

    sid_product = SeaIceDriftProducts(hem=hem, product_id=config['options']['ice_drift_product'], out_epsg=out_epsg)
    sid_product.get_file_list(config['auxiliary']['ice_drift'][config['options']['ice_drift_product']])
    sid_product.get_file_dates()

    t2m_product = AirTemperatureProducts(hem=hem, product_id=config['options']['t2m_product'], out_epsg=out_epsg)
    t2m_product.get_file_list(config['auxiliary']['t2m'][config['options']['t2m_product']])
    t2m_product.get_file_dates()

    thermo_model = config['options']['proc_step_options']['stacking']['thermo_change']['model']
    #if thermo_model=='None' : thermo_model = None
    if type(config['options']['proc_step_options']['stacking']['thermo_change']['oce_heat_flux']) is not int:
        logger.info('Need to be implemented with a reanalysis')
        ohf_product = OceanHeatFluxProducts(hem=hem, product_id=config['options']['ohf_product'], out_epsg=out_epsg)
        ohf_product.get_file_list(config['auxiliary']['ohf'][config['options']['ohf_product']])
        ohf_product.get_file_dates()

    else:
        ohf_product = config['options']['proc_step_options']['stacking']['thermo_change']['oce_heat_flux']
        logger.info(f"The Ocean heat flux used is a constant : {config['options']['proc_step_options']['stacking']['thermo_change']['oce_heat_flux']}")

    if direct == 'f':
        d_sgn = 1
        d_sgn_drift = 1
        d_sgn_t2m = 0
        # After the source period, parcels are still advected but no new SIT is read.
        day_range = range(0, tracking_length, d_sgn)
        source_index_offset = 0
    else:
        d_sgn = -1
        d_sgn_drift = 0
        d_sgn_t2m = -1
        # Give days before t_start non-negative indices in the stack structure.
        day_range = range(source_length - 1, -extension_length - 1, d_sgn)
        source_index_offset = extension_length

    # initialize drift aware processor
    processor = DriftAwareProcessor(sit_product, master=master, scheme=scheme, grid=grid)

    # for each day we want to stack
    for i in day_range:

        processor.i = i + source_index_offset
        t0 = stk_opt['t_start'] + datetime.timedelta(days=i) 
        t1 = stk_opt['t_start'] + datetime.timedelta(days=i + 1) 
        has_source_sit = 0 <= i < source_length
        active_thermo_model = thermo_model if has_source_sit else None
        if has_source_sit:
            sit_product.get_target_files(t0, t1)
            sic_product.target_files = sic_product.get_target_files(t0, t1)

            # Number of empty list for missions
            empty_lists = [k for k, v in sit_product.target_files.items() if isinstance(v, list) and len(v) == 0]
            sit_product.target_files = {k: (None if isinstance(v, list) and len(v) == 0 else v) for k, v in (sit_product.target_files or {}).items()}
            sensor_k = [s for s in sensor if sit_product.target_files.get(s) is not None]
            file_counts = {k: len(v) for k, v in sit_product.target_files.items() if isinstance(v, list) and len(v) > 0}
            # Build the baseline so the line that corresponds to the actual time, without any advection needed.
            if len(empty_lists) >= len(sensor_k):
                logger.warning(t0.strftime("%Y%m%d") + ': Missing sea ice thickness files for: ' + str(empty_lists) + '. Skipping this date.')
            if (len(empty_lists) < len(sensor_k)) and sic_product.target_files:
                logger.info(t0.strftime("%Y%m%d") + ': altimetry files (n): ' + str(file_counts))
                logger.info(t0.strftime("%Y%m%d") + ': ice_conc file day0: ' + os.path.basename(sic_product.target_files))
                sit_product.get_product(sensor_k)
                n_observations = len(sit_product.product)
                sit_product.product = filter_sit_by_spatial_filter(
                    sit_product.product,
                    bounds=lon_lat_bounds,
                    polygon=lon_lat_polygon,
                    vector_file=spatial_filter_file,
                )
                if any(value is not None for value in (lon_lat_bounds, lon_lat_polygon, spatial_filter_file)):
                    logger.info(
                        f"{t0:%Y%m%d}: kept {len(sit_product.product)}/{n_observations} SIT observations "
                        "inside the configured spatial filter."
                    )
                sic_product.ice_conc = sic_product.get_ice_concentration(sic_product.target_files)

            #if ICESAT-2, then we need to filter out the total freeboard (snow depth purposes)
            if (len(sensor_k) == 1) and (sensor_k[0] == 'icesat2'):
                if 'tFB_clim' in config['auxiliary']:

                    sit_clim_product = SeaIceThicknessClimProducts(hem=hem, product_id='tFB_clim',
                                                            out_epsg=out_epsg)
                    logger.info(f"Climatology is used to filter out MIZ outliers from : {config['auxiliary']['tFB_clim']}")
                    sit_clim_product.get_file_list(config['auxiliary']['tFB_clim'])
                    sit_clim_product.get_file_dates()
                    
                    sit_clim_product.target_files = sit_clim_product.get_target_files(t0, t1)

                    sit_clim_product.sit_clim = sit_clim_product.get_tFB_clim(sit_clim_product.target_files)
                #keep only the total freeboard with quality flag <= 2
                sit_product.product = sit_product.product[(sit_product.product['total_freeboard_quality_flag'] <= 4) & (sit_product.product['total_freeboard_quality_flag'] >= 0)]
                #sit_product.product = compute_apply_flag(sit_product, sic_product, sit_clim_product)    
            elif (sensor_k[0] != 'icesat2') and (target_var == 'sea_ice_thickness'):
                if 'SIT_clim' in config['auxiliary']:

                    sit_clim_product = SeaIceThicknessClimProducts(hem=hem, product_id='sit_clim',
                                                            out_epsg=out_epsg)
                    logger.info(f"Climatology is used to filter out MIZ outliers from : {config['auxiliary']['SIT_clim']}")
                    sit_clim_product.get_file_list(config['auxiliary']['sit_clim'])
                    sit_clim_product.get_file_dates()
                    
                    sit_clim_product.target_files = sit_clim_product.get_target_files(t0, t1)

                    sit_clim_product.sit_clim = sit_clim_product.get_SIT_clim(sit_clim_product.target_files)
                
            if not sit_product.product.empty:
                processor.baseline_proc(sic_product, hist_n_bins, hist_range, sit_clim = sit_clim_product if 'sit_clim_product' in locals() else None)
            else:
                logger.warning(t0.strftime("%Y%m%d") + ': No SIT observations remain after lon/lat filtering.')
        else:
            logger.info(t0.strftime("%Y%m%d") + ': continuing tracking without loading sea ice thickness.')
        
        # The sea ice concentration is taken at t1 check data after beeing advected
        sic_product.target_files = sic_product.get_target_files(t0 + d_sgn * dt1d, t1 + d_sgn * dt1d)
        # The sea ice drift to advect parcel at t0 is the one referenced as t1
        # Indeed the reference correspond to the end of the 24h data range that cover each file
        sid_product.target_files = sid_product.get_target_files(t0 + d_sgn_drift * dt1d, t1 + d_sgn_drift * dt1d)
        if active_thermo_model is not None:
            t2m_product.target_files = t2m_product.get_target_files(t0 + d_sgn_t2m * dt1d, t1 + d_sgn_t2m * dt1d)
        else:
            t2m_product.target_files = None
        if type(ohf_product) is not int and active_thermo_model is not None:
            ohf_product.target_files = ohf_product.get_target_files(t0 + d_sgn_t2m * dt1d, t1 + d_sgn_t2m * dt1d)

        if sic_product.target_files and sid_product.target_files and (t2m_product.target_files or not active_thermo_model) :
            logger.info(t0.strftime("%Y%m%d") + ': ice_conc file day'+str(d_sgn)+': ' +
                        os.path.basename(sic_product.target_files))
            logger.info(t0.strftime("%Y%m%d") + ': ice_drift file: ' +
                        os.path.basename(sid_product.target_files))
            
            sic_product.ice_conc_ahead = sic_product.get_ice_concentration(sic_product.target_files)
            sid_product.get_ice_drift(sid_product.target_files, sic_product.ice_conc_ahead)
            
            if active_thermo_model:
                logger.info(t0.strftime("%Y%m%d") + ': t2m file: ' +
                        os.path.basename(t2m_product.target_files))
                t2m_product.get_air_temperature(t2m_product.target_files)
            
            if type(ohf_product) is not int and active_thermo_model is not None:
                ohf_product.get_ocean_heat_flux(ohf_product.target_files)
                logger.info(t0.strftime("%Y%m%d") + ': ohf file: ' +
                        os.path.basename(ohf_product.target_files))
                
            # check if the date is still in the range
            if (d_sgn == -1 and processor.i > 0) or (d_sgn == 1 and processor.i < tracking_length - 1):
                m = processor.drift_aware_proc(sid_product, sic_product, t2m_product, ohf_product, stk_opt['t_window'], d_sgn, tracking_length - 1, active_thermo_model)

        is_terminal_day = (
            processor.i == tracking_length - 1 if direct == 'f' else processor.i == 0
        )
        gdf_final = processor.concat_gdfs(processor.i, m)
        if save_only_terminal_file and not is_terminal_day:
            continue
        gdf_final[target_var+'_drift_unc'] = np.sqrt(gdf_final[target_var+'_drift_unc'])
        gdf_final = gdf_final.drop(columns=['xu', 'yu'])
        gdf_final["geometry"] = gdf_final["geometry"].apply(lambda gdf: MultiPoint(gdf))
        gdf_final = gpd.GeoDataFrame(gdf_final, geometry='geometry')
        gdf_final['divergence'] = gdf_final['divergence'].apply(json.dumps)
        gdf_final['shear'] = gdf_final['shear'].apply(json.dumps)
        if thermo_model:
            gdf_final['rate_thermo_change_mod'] = gdf_final.apply(lambda row: row['thermo_change_mod'] / abs(row['dt_days']) if row['dt_days'] != 0 else 0, axis=1)
            gdf_final['rate_thermo_growth_mod'] = gdf_final.apply(lambda row: row['thermo_growth_mod'] / abs(row['dt_days']) if row['dt_days'] != 0 else 0, axis=1)
            """
            gdf_final['rate_thermo_change_mod2'] = gdf_final.apply(lambda row: row['thermo_change_mod2'] / abs(row['dt_days']) if row['dt_days'] != 0 else 0, axis=1)
            gdf_final['rate_thermo_growth_mod2'] = gdf_final.apply(lambda row: row['thermo_growth_mod2'] / abs(row['dt_days']) if row['dt_days'] != 0 else 0, axis=1)

            gdf_final['rate_thermo_change_mod3'] = gdf_final.apply(lambda row: row['thermo_change_mod3'] / abs(row['dt_days']) if row['dt_days'] != 0 else 0, axis=1)
            gdf_final['rate_thermo_growth_mod3'] = gdf_final.apply(lambda row: row['thermo_growth_mod3'] / abs(row['dt_days']) if row['dt_days'] != 0 else 0, axis=1)

            gdf_final['rate_thermo_change_mod4'] = gdf_final.apply(lambda row: row['thermo_change_mod4'] / abs(row['dt_days']) if row['dt_days'] != 0 else 0, axis=1)
            gdf_final['rate_thermo_growth_mod4'] = gdf_final.apply(lambda row: row['thermo_growth_mod4'] / abs(row['dt_days']) if row['dt_days'] != 0 else 0, axis=1)

            gdf_final['rate_thermo_change_mod5'] = gdf_final.apply(lambda row: row['thermo_change_mod5'] / abs(row['dt_days']) if row['dt_days'] != 0 else 0, axis=1)
            gdf_final['rate_thermo_growth_mod5'] = gdf_final.apply(lambda row: row['thermo_growth_mod5'] / abs(row['dt_days']) if row['dt_days'] != 0 else 0, axis=1)

            gdf_final['rate_thermo_change_mod6'] = gdf_final.apply(lambda row: row['thermo_change_mod6'] / abs(row['dt_days']) if row['dt_days'] != 0 else 0, axis=1)
            gdf_final['rate_thermo_growth_mod6'] = gdf_final.apply(lambda row: row['thermo_growth_mod6'] / abs(row['dt_days']) if row['dt_days'] != 0 else 0, axis=1)
            """
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
    stk_opt = config['options']['proc_step_options']['stacking']
    if stk_opt.get('continue_tracking_without_sit', False) and stk_opt['mode'] not in ('f', 'r'):
        raise ValueError(
            "continue_tracking_without_sit requires a single direction: set stacking.mode to 'f' or 'r'."
        )
    if stk_opt.get('save_only_terminal_file', False):
        if stk_opt['mode'] not in ('f', 'r'):
            raise ValueError("save_only_terminal_file requires stacking.mode to be 'f' or 'r'.")
        if stk_opt['t_window'] < stk_opt['t_length']:
            raise ValueError(
                "save_only_terminal_file requires t_window >= t_length so the terminal file contains all source SIT."
            )
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
                pool.apply_async(stack_proc, args=(config, mode, grid))
            pool.close()
            pool.join()
        else:
            for mode in stk_opt['mode']:
                stack_proc(config, mode, grid)

        logger.info('start merging forward and reverse stacks')

        list_f = sorted(glob.glob(os.path.join(config['output_dir']['trajectories'], f'*_F-*.csv')))
        list_r = sorted(glob.glob(os.path.join(config['output_dir']['trajectories'], f'*_R-*.csv')))
        extension_length = (
            max(0, stk_opt['t_window'] - stk_opt['t_length'])
            if stk_opt.get('continue_tracking_without_sit', False) else 0
        )
        if stk_opt.get('save_only_terminal_file', False):
            terminal_day = stk_opt['t_length'] + extension_length - 1 if stk_opt['mode'] == 'f' else -extension_length
            merge_day_range = range(terminal_day, terminal_day + 1)
        elif stk_opt['mode'] == 'f':
            merge_day_range = range(0, stk_opt['t_length'] + extension_length)
        else:
            merge_day_range = range(-extension_length, stk_opt['t_length'])
        if multiproc:
            pool = mp.Pool(stk_opt['num_cpus'])
            for j in merge_day_range:
                pool.apply_async(
                    merge_forward_reverse_stacks, args=(
                        config, growth_grid, growth_cell_width, cell_width, list_f, list_r, j))
            pool.close()
            pool.join()
        else:
            for j in merge_day_range:
                merge_forward_reverse_stacks(config, growth_grid, growth_cell_width, cell_width, list_f, list_r, j)
