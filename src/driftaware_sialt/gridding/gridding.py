import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import datetime
import glob
import re
import os
import shutil
import sys
import multiprocessing as mp
from driftaware_sialt.io_tools import transform_coords
from driftaware_sialt.io_tools import get_sea_ice_regions
from driftaware_sialt.io_tools import create_out_dir
from driftaware_sialt.io_tools import read_dasit_csv
from driftaware_sialt.io_tools import read_dasit_metadata
from driftaware_sialt.gridding.prepare_netcdf import PrepareNetcdf
from driftaware_sialt.gridding import gridding_lib
from driftaware_sialt.products.sea_ice_concentration import SeaIceConcentrationProducts
from driftaware_sialt.products.selection import select_product
from loguru import logger
from driftaware_sialt.io_tools import init_logger

def organize_files_by_date(source_dir, target_dir):
    files = [f for f in os.listdir(source_dir) if f.endswith('.nc')]
    for file in files:
        if os.path.basename(file).startswith("._"):
            os.remove(os.path.join(source_dir, file))
            continue
        match = re.search(r'(\d{8})', file)
        if match:
            date_str = match.group(1)
            year, month = date_str[:4], date_str[4:6]
            year_dir = os.path.join(target_dir, year)
            month_dir = os.path.join(year_dir, month)
            os.makedirs(month_dir, exist_ok=True)
            src_path = os.path.join(source_dir, file)
            dest_path = os.path.join(month_dir, file)
            shutil.move(src_path, dest_path)
        else:
            logger.info(f"Date not found in file name: {file}")
    if not os.listdir(source_dir):
        os.rmdir(source_dir)
    else:
        logger.info(f"Source directory {source_dir} is not empty and has not been removed.")


def get_deformation(row):
    mean_tot = [np.linalg.norm([a, b]) for a, b in zip(row['shear'], row['divergence'])]
    return np.mean(mean_tot)


def get_row_mean(row):
    return np.mean(row)


def add_full_domain_ice_concentration(dataset, config, time_center):
    """Sample the daily SIC product independently of thickness observations."""
    products = SeaIceConcentrationProducts.load_products(
        config['options']['ice_conc_products'],
        config['auxiliary']['ice_conc'],
        hem=config['options']['hemisphere'],
        out_epsg=config['options']['out_epsg'])
    product = select_product(
        products, time_center.replace(hour=0),
        time_center.replace(hour=0) + datetime.timedelta(days=1))
    if product is None:
        logger.warning(
            f"{time_center:%Y%m%d}: no nearby concentration file; "
            "retaining trajectory-sampled concentration")
        return dataset

    ice_conc = product.get_ice_concentration(product.target_files)
    xc, yc = np.meshgrid(dataset.xc.values, dataset.yc.values)
    values = product.interp_ice_concentration(
        ice_conc, xc.ravel(), yc.ravel()).reshape(xc.shape) * 100.0
    values[(values <= 0) | (values > 100)] = np.nan
    dataset['sea_ice_concentration'] = (
        ('time', 'yc', 'xc'), values[np.newaxis, :, :])
    return dataset


def get_source_stack_metadata(config, file_list):
    """Return common stack metadata and reject incompatible trajectory files."""
    fallback = config['gridding'].get('source_stack')
    expected_target = config['options']['target_variable']
    source_stack = None

    for file in file_list:
        metadata = read_dasit_metadata(file)
        file_source_stack = metadata.get('source_stack', fallback)
        if file_source_stack is None:
            raise ValueError(
                f'{file} uses the legacy CRS-only header. Regenerate the trajectory '
                'CSV with format_version 1 or provide source_stack in the gridding '
                'configuration as a temporary compatibility fallback.')

        file_target = metadata.get('target_variable')
        if file_target and file_target != expected_target:
            raise ValueError(
                f'trajectory target_variable {file_target!r} in {file} does not '
                f'match configured target_variable {expected_target!r}')

        if source_stack is None:
            source_stack = file_source_stack
        elif file_source_stack != source_stack:
            raise ValueError(
                f'incompatible source_stack metadata found in trajectory file: {file}')

    return source_stack


def process_file(config, file_list, grid, region_grid, region_metadata, source_stack):
    init_logger(config)
    target_var = config['options']['target_variable']
    out_epsg = config["options"]["out_epsg"]
    grd_opt = config['gridding']
    # declare histogram options
    hist_n_bins = source_stack['histogram']['n_bins']
    hist_range = source_stack['histogram']['range']["freeboard" if "freeboard" in target_var else "thickness"]
    hist_bin_size = (hist_range[1] - hist_range[0]) / hist_n_bins

    # declare gridding options
    gridding_mode = grd_opt['mode']
    dt_days_max = grd_opt['dt_days_max']
    var_range = grd_opt['target_variable_range']["freeboard" if "freeboard" in target_var else "thickness"]
    out_dir = config['output_dir']['gridded_data']
    is_weight = grd_opt['weighting']['is_weight']
    weight_var = grd_opt['weighting']['var_to_weight_with']

    source_products = {}
    for i, file in enumerate(file_list):
        logger.info('process csv file: ' + os.path.basename(file))
        data_tmp, metadata = read_dasit_csv(file, return_metadata=True)
        for source_type, products in metadata.get(
                'source_products', {}).items():
            target = source_products.setdefault(source_type, [])
            target.extend(product for product in products if product not in target)
        if i == 0:
            data = data_tmp
        else:
            data = pd.concat([data, data_tmp], ignore_index=True)
            

    traj_geom = data['geometry']
    start_location = data["geometry"].apply(lambda g: g.geoms[0])
    target_location = data["geometry"].apply(lambda g: g.geoms[-1])
    data["geometry"] = target_location
    data.to_crs(crs=out_epsg, inplace=True)
    data = data[data['dt_days'].abs() <= dt_days_max]
    if data.empty:
        logger.warning(f"No data within the specified dt_days_max of {dt_days_max} days in file: {file}")
        return  
    data['dist_acquisition'] = start_location.distance(target_location) / 1000.0
    data['divergence'] = data['divergence'].apply(lambda x: [float(val) for val in x.split()])
    data['dynamic_change_rate_tmp'] = data['divergence'].apply(lambda x: [np.exp(-val) for val in x])
    # Handle both old format (without _uncorrected) and new format (with _uncorrected)
    uncorrected_col = f"{target_var}_uncorrected" if f"{target_var}_uncorrected" in data.columns else target_var
    if uncorrected_col in data.columns:
        data['dynamic_change_rate'] = data.apply(lambda row: [-row[uncorrected_col] * val for val in row["divergence"]], axis=1)
    else:
        logger.warning(f"Column '{uncorrected_col}' not found. Skipping dynamic_change_rate computation.")
    if 'growth_interpolated' in data.columns:
        data['thermo_change_rate'] = data.apply(lambda row: [row["growth_interpolated"] - val for val in row["dynamic_change_rate"]], axis=1)

    #data['thermo_change_rate'] = data.apply(lambda row: row["growth_interpolated"] - row["dynamic_change_rate"], axis=1)
    data['shear'] = data['shear'].apply(lambda x: [float(val) for val in x.split()])

    if gridding_mode == 'da':
        data["geometry"] = target_location
    elif gridding_mode == 'cv':
        data["geometry"] = start_location
    else:
        logger.error('Gridding mode does not exist: %s', gridding_mode)
        sys.exit()

    data['ice_conc'] = data['ice_conc'] * 100.0
    # Only apply target_var filtering if the column exists
    if target_var in data.columns:
        data[(data[target_var] > var_range[1]) |
             (data[target_var] < var_range[0])] = np.nan
    else:
        logger.warning(f"Column '{target_var}' not found in data. Skipping target variable filtering.")
    if 'clim_interp' in data.columns:
        data = data.dropna(subset=data.columns.difference(['growth', 'longitude', 'latitude']))
    else:
        data = data.dropna(subset=data.columns.difference(['growth']))
    data = data.reset_index()
    time_center = datetime.datetime.strptime(
        re.split('-', os.path.basename(file))[-2], '%Y%m%d') + datetime.timedelta(hours=12)
    # extract histogram
    data_hist = data[target_var + '_hist'].str.split(expand=True).astype(int)
    data_hist.columns = np.arange(hist_n_bins).astype(str)
    data_hist = gpd.GeoDataFrame(pd.concat([data_hist, data.geometry, data.dt_days], axis=1))

    # add modal value to main data frame
    data[target_var + '_mode'] = gridding_lib.modal_var(data_hist, hist_n_bins, hist_bin_size, hist_range)

    merged = gpd.sjoin(data, grid, how='left', predicate='within')
    merged_hist = gpd.sjoin(data_hist, grid, how='left', predicate='within')

    geo = merged.groupby(['index_right', 'dt_days'], as_index=False).first()['geometry']

    tmp_hist_grouped = merged_hist.drop(columns='geometry').groupby(['index_right', 'dt_days'], as_index=False).sum()
    tmp_hist_grouped['geometry'] = geo
    tmp_hist = gpd.GeoDataFrame(tmp_hist_grouped, geometry='geometry')

    tmp_hist = gpd.GeoDataFrame(tmp_hist)

    tmp_hist_grid = gridding_lib.grid_data(tmp_hist, grid,
                                           np.arange(hist_n_bins).astype('str').tolist(),
                                           np.arange(hist_n_bins).astype('str').tolist(),
                                           hist_n_bins,
                                           hist_range,
                                           fill_nan=True,
                                           agg_mode=['sum'])
    if target_var + 'total_unc' not in data.columns:
        if 'freeboard' not in target_var:
            data[target_var+'_total_unc'] = np.sqrt(data[target_var+'_growth_unc']**2 +
                                                    data[target_var+'_drift_unc']**2 +
                                                    data[target_var+'_l2_unc']**2)
        else:
            data[target_var+'_total_unc'] = np.sqrt(data[target_var+'_drift_unc']**2 +
                                                    data[target_var+'_l2_unc']**2)

    data['deformation'] = data.apply(get_deformation, axis=1)
    data['divergence'] = data["divergence"].apply(get_row_mean)
    data['shear'] = data["shear"].apply(get_row_mean)
    data['dynamic_change_rate'] = data['dynamic_change_rate'].apply(get_row_mean)
    if 'thermo_change_rate' in data.columns:
        data['thermo_change_rate'] = data['thermo_change_rate'].apply(get_row_mean)

    prepare_netcdf = PrepareNetcdf(
        config, file, region_grid, region_metadata, source_stack,
        source_products)
    var, var_rename = prepare_netcdf.select_variables(data)
    if is_weight:
        master = gridding_lib.grid_data(data, grid, var, var_rename, fill_nan=True, agg_mode=['weighted_mean'], weight_var=weight_var)
        # Take into account the weight for the uncertainty computation
        #master[target_var + '_total_unc'] = master_unc[target_var + '_total_unc'].copy()
        #master[target_var + '_growth_unc'] = master_unc[target_var + '_growth_unc'].copy()
        #master[target_var + '_drift_unc'] = master_unc[target_var + '_drift_unc'].copy()
        #master[target_var + '_l2_unc'] = master_unc[target_var + '_l2_unc'].copy()
    else:
        master = gridding_lib.grid_data(data, grid, var, var_rename, fill_nan=True, agg_mode=['mean'])

    
    if 'cryosat2_cnt' in merged.columns:
        master['cryosat2_cnt'] = gridding_lib.grid_data(data, grid, ['cryosat2_cnt'], ['cryosat2_cnt'], fill_nan=True, agg_mode=['sum'])['cryosat2_cnt_sum']
    if 'sentinel3a_cnt' in merged.columns:
        master['sentinel3a_cnt'] = gridding_lib.grid_data(data, grid, ['sentinel3a_cnt'], ['sentinel3a_cnt'], fill_nan=True, agg_mode=['sum'])['sentinel3a_cnt_sum']
    if 'sentinel3b_cnt' in merged.columns:
        master['sentinel3b_cnt'] = gridding_lib.grid_data(data, grid, ['sentinel3b_cnt'], ['sentinel3b_cnt'], fill_nan=True, agg_mode=['sum'])['sentinel3b_cnt_sum']
    
    master[target_var + '_std'] = gridding_lib.grid_data(
        data, grid, [target_var], [target_var], fill_nan=True, agg_mode=['std'])[target_var + '_std']
    master = master.join(tmp_hist_grid.drop(columns=['geometry']))
    centroidseries = master['geometry'].centroid
    master['xc'], master['yc'] = round(centroidseries.x), round(centroidseries.y)
    master['time'] = (time_center - datetime.datetime(1970, 1, 1, 0, 0)).total_seconds()
    master['longitude'], master['latitude'] = transform_coords(master['xc'], master['yc'], out_epsg, 'EPSG:4326')
    master = master.set_index(['time', 'yc', 'xc'])
    master.drop(columns=['geometry'], inplace=True)
    master = xr.Dataset.from_dataframe(master)
    master = master.reindex(yc=list(reversed(master.yc)))
    master = add_full_domain_ice_concentration(master, config, time_center)
    master = master.set_coords(("longitude", "latitude"))
    master = prepare_netcdf.add_projection_field(master)
    master = prepare_netcdf.add_time_bnds(master, data['t0'].min(), data['t0'].max())
    master = prepare_netcdf.add_region_code(master)
    master = prepare_netcdf.add_histogram(master)
    master = prepare_netcdf.set_var_attrbs(master)
    master = prepare_netcdf.set_glob_attrbs(master)
    outfile = prepare_netcdf.make_netcdf_filename(grid, gridding_mode)
    comp = dict(zlib=True, complevel=1)
    encoding = {var: comp for var in master.data_vars}
    master.to_netcdf(path=os.path.join(out_dir, outfile), encoding=encoding, format="NETCDF4")
    logger.info('generated netcdf file: ' + outfile)


def gridding(config):
    sensor = config['options']['sensor']
    hemisphere = config['options']['hemisphere']
    grd_opt = config['gridding']
    netcdf_bounds = grd_opt['netcdf_grid']['bounds']
    if grd_opt['csv_dir'] == "all":
        file_list = sorted([os.path.join(root, file)
                            for root, _, files in os.walk(config['output_dir']['trajectories'])
                            for file in files
                            if file.endswith('.csv')])
    else:
        csv_dir = os.path.join(config['output_dir']['trajectories'], grd_opt['csv_dir'])
        file_list = sorted(glob.glob(os.path.join(csv_dir,'*.csv')))

    if not file_list:
        raise FileNotFoundError('no trajectory CSV files found for gridding')

    source_stack = get_source_stack_metadata(config, file_list)

    grid, cell_width = gridding_lib.define_grid(
        netcdf_bounds,
        grd_opt['netcdf_grid']['dim'],
        config['options']['out_epsg'],
        grid_type='circular')

    config['output_dir']['gridded_data'] = create_out_dir(
        config, config['output_dir']['gridded_data'], cell_width, source_stack)
    region_grid, region_metadata = get_sea_ice_regions(
        config['auxiliary']['reg_mask'][hemisphere],
        netcdf_bounds,
        round(0.5 * np.sqrt(2) * cell_width),
        config['options']['out_epsg'],
        hemisphere)

    date_pattern = re.compile(r"\b(20\d{6})\b")

    grouped_files = []

    while file_list:
        file = file_list.pop(0)  
        match = date_pattern.search(file)
        
        if match:
            date = match.group(1)
            sublist = [file]  
            
            remaining_files = []
            for other_file in file_list:
                if date_pattern.search(other_file) and date_pattern.search(other_file).group(1) == date:
                    sublist.append(other_file)
                else:
                    remaining_files.append(other_file)

            grouped_files.append(sublist)
            file_list = remaining_files

    if grd_opt['multiproc']:
        logger.info('start multiprocessing')
        pool = mp.Pool(grd_opt['num_cpus'])
        for i in range(len(grouped_files)):
            pool.apply_async(
                process_file,
                args=(config, grouped_files[i], grid, region_grid,
                      region_metadata, source_stack))
        pool.close()
        pool.join()
    else:
        for i in range(len(grouped_files)):
            process_file(
                config, grouped_files[i], grid, region_grid,
                region_metadata, source_stack)

    if grd_opt["organize_files"]:
        organize_files_by_date(config['output_dir']['gridded_data'],
                               os.path.dirname(config['output_dir']['gridded_data']))
