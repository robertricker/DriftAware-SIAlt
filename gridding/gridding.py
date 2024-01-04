import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import datetime
import glob
import re
import os
import sys
import multiprocessing as mp
from io_tools import transform_coords
from io_tools import get_sea_ice_regions
from io_tools import create_out_dir
from gridding.prepare_netcdf import PrepareNetcdf
from gridding import gridding_lib
from loguru import logger
from io_tools import init_logger


def get_deformation(row):
    mean_tot = [np.linalg.norm([a, b]) for a, b in zip(row['shear'], row['divergence'])]
    return np.mean(mean_tot)


def get_row_mean(row):
    return np.mean(row)


def process_file(config, file, grid, region_grid):
    init_logger(config)
    sensor = config['options']['sensor']
    target_var = config['options']['target_variable']
    out_epsg = config["options"]["out_epsg"]
    stk_opt = config['options']['proc_step_options']['stacking']
    grd_opt = config['options']['proc_step_options']['gridding']

    # declare histogram options
    hist_n_bins = stk_opt['hist']['n_bins']
    hist_range = stk_opt['hist']['range']["freeboard" if "freeboard" in target_var else "thickness"]
    hist_bin_size = (hist_range[1] - hist_range[0]) / hist_n_bins

    # declare gridding options
    gridding_mode = grd_opt['mode']
    var_range = grd_opt['target_variable_range']["freeboard" if "freeboard" in target_var else "thickness"]
    out_dir = config['dir'][sensor]['netcdf']

    logger.info('process geosjson file: ' + os.path.basename(file))

    data = gpd.read_file(file)
    data.crs = "epsg:" + re.findall(r'epsg(\d{4})', os.path.basename(file))[0]
    data.to_crs(crs=out_epsg, inplace=True)
    start_location = data["geometry"].apply(lambda g: g.geoms[0])
    target_location = data["geometry"].apply(lambda g: g.geoms[-1])
    data['dist_acquisition'] = start_location.distance(target_location) / 1000.0
    data['divergence'] = data['divergence'].apply(lambda x: [float(val) for val in x.split()])
    data['shear'] = data['shear'].apply(lambda x: [float(val) for val in x.split()])

    gridding_modes = {
        'drift-aware': ('daware', target_location),
        'conventional': ('conv', start_location)}

    if gridding_mode in gridding_modes:
        file_prefix, data["geometry"] = gridding_modes[gridding_mode]
    else:
        logger.error('Gridding mode does not exist: %s', gridding_mode)
        sys.exit()
    data['ice_conc'] = data['ice_conc'] * 100.0
    data[(data[target_var] > var_range[1]) |
         (data[target_var] < var_range[0])] = np.nan
    data = data.dropna(subset=data.columns.difference(['growth']))
    data = data.reset_index()
    time_bnds = np.array([[data['t0'].min(), data['t0'].max()]])
    time_center = datetime.datetime.strptime(
        re.split('-', os.path.basename(file))[3], '%Y%m%d') + datetime.timedelta(hours=12)
    # extract histogram
    data_hist = data[target_var + '_hist'].str.split(expand=True).astype(int)
    data_hist.columns = np.arange(hist_n_bins).astype(str)
    data_hist = gpd.GeoDataFrame(pd.concat([data_hist, data.geometry, data.dt_days], axis=1))

    # add modal value to main data frame
    zc, data[target_var + '_mode'] = gridding_lib.modal_var(data_hist, hist_n_bins, hist_bin_size, hist_range)

    merged = gpd.sjoin(data, grid, how='left', op='within')
    merged_hist = gpd.sjoin(data_hist, grid, how='left', op='within')

    geo = merged.groupby(['index_right', 'dt_days'], as_index=False).first()['geometry']
    tmp_hist = merged_hist.groupby(['index_right', 'dt_days'], as_index=False).sum().assign(geometry=geo)
    tmp_hist = gpd.GeoDataFrame(tmp_hist)

    tmp_hist_grid = gridding_lib.grid_data(tmp_hist, grid,
                                           np.arange(hist_n_bins).astype('str').tolist(),
                                           np.arange(hist_n_bins).astype('str').tolist(),
                                           hist_n_bins,
                                           hist_range,
                                           fill_nan=True,
                                           agg_mode=['sum'])

    data[target_var+'_total_unc'] = np.sqrt(data[target_var+'_growth_unc']**2 +
                                            data[target_var+'_drift_unc']**2 +
                                            data[target_var+'_unc']**2)

    data['deformation'] = data.apply(get_deformation, axis=1)
    data['divergence'] = data["divergence"].apply(get_row_mean)
    data['shear'] = data["shear"].apply(get_row_mean)
    prepare_netcdf = PrepareNetcdf(config)
    var, var_rename = prepare_netcdf.set_field_names()
    master = gridding_lib.grid_data(data, grid, var, var_rename, fill_nan=True)
    master = master.join(tmp_hist_grid.drop(columns=['geometry']))
    master = prepare_netcdf.drop_fields(master)
    centroidseries = master['geometry'].centroid
    master['yc'], master['xc'] = centroidseries.x, centroidseries.y
    master['time'] = (time_center - datetime.datetime(1970, 1, 1, 0, 0)).total_seconds()
    master['longitude'], master['latitude'] = transform_coords(master['yc'], master['xc'], out_epsg, 'EPSG:4326')
    master = master.set_index(['time', 'xc', 'yc'])
    master.drop(columns=['geometry'], inplace=True)
    master = xr.Dataset.from_dataframe(master)
    master = master.set_coords(("longitude", "latitude"))
    master = prepare_netcdf.add_projection_field(master)
    master["time_bnds"] = xr.DataArray(time_bnds, dims=("time", "nv"), coords={"time": master.time})
    master['region_flag'] = (['time', 'xc', 'yc'], region_grid[np.newaxis, :, :])
    hist_arr = xr.concat([master[str(i) + '_sum'] for i in np.arange(hist_n_bins)], dim='zc').values
    master = master.drop_vars([str(i) + '_sum' for i in np.arange(hist_n_bins)])

    master = master.assign_coords(zc=("zc", zc))
    master[target_var + '_hist'] = (['time', 'xc', 'yc', 'zc'], np.transpose(hist_arr, (1, 2, 3, 0)))

    master = prepare_netcdf.set_var_attrbs(master)
    master = prepare_netcdf.set_glob_attrbs(master)

    outfile = (re.split('-', os.path.basename(file))[0] + "-" +
               re.split('-', os.path.basename(file))[1] + "-" +
               re.split('-', os.path.basename(file))[2] + "-" +
               re.split('-', os.path.basename(file))[3] + "-" +
               re.split('-', os.path.basename(file))[4] + "-" +
               'epsg' + out_epsg.split(":")[1] + "_" +
               "{:.0f}".format(grid.geometry[0].length / 4 / 100.0) + '.nc')

    comp = dict(zlib=True, complevel=1)
    encoding = {var: comp for var in master.data_vars}
    master.to_netcdf(path=out_dir + file_prefix + '-' + outfile, encoding=encoding, format="NETCDF4")
    logger.info('generated netcdf file: ' + outfile)


def gridding(config):

    # declare sensor and target variable options
    sensor = config['options']['sensor']
    target_var = config['options']['target_variable']
    grd_opt = config['options']['proc_step_options']['gridding']
    multiproc = grd_opt['multiproc']
    netcdf_bounds = grd_opt['netcdf_grid']['bounds']
    geojson_dir = config['dir'][sensor]['geojson'] + grd_opt['geojson_dir']
    file_list = sorted(glob.glob(geojson_dir + "/" + target_var + '*.geojson'))

    grid, cell_width = gridding_lib.define_grid(
        netcdf_bounds,
        grd_opt['netcdf_grid']['dim'],
        config['options']['out_epsg'])

    config['dir'][sensor]['netcdf'] = create_out_dir(config, config['dir'][sensor]['netcdf'], cell_width)
    region_grid = get_sea_ice_regions(config['dir']['auxiliary']['reg_mask'], netcdf_bounds, cell_width,
                                      config['options']['out_epsg'])

    if multiproc:
        logger.info('start multiprocessing')
        pool = mp.Pool()
        for file in file_list:
            pool.apply_async(process_file, args=(config, file, grid, region_grid))
        pool.close()
        pool.join()
    else:
        for file in file_list:
            process_file(config, file, grid, region_grid)
