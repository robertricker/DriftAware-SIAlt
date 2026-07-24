import datetime
import glob
import os
import re

import pandas as pd
import xarray as xr
from loguru import logger

from driftaware_sialt.products.sea_ice_concentration import (
    SeaIceConcentrationProducts,
)
from driftaware_sialt.products.selection import select_product
from driftaware_sialt.volume.calculations import calculate_volume_and_mass
from driftaware_sialt.volume.output import (
    build_regional_summary,
    organize_files_by_date,
    write_volume_dataset,
)


def run(config):
    hem = config["options"]["hemisphere"]
    out_epsg = config["options"]["out_epsg"]
    volume_opt = config['volume']
    which_ice_conc = volume_opt['ice_conc']
    target_var = config['options']['target_variable']

    config['output_dir']['volume_data'] = config['output_dir']['volume'] + '/' + volume_opt['sub_dir']
    file_list = sorted(glob.glob(os.path.join(config['output_dir']['gridded_data'] + volume_opt['sub_dir'], '**', '*.nc'), recursive=True))
    if len(file_list) == 0:
        logger.error('No NetCDF files found for volume computation in: %s', os.path.join(config['output_dir']['gridded_data'] + volume_opt['sub_dir'], '**', '*.nc'))
        return
    out_dir = config['output_dir']['volume']
    si_density_param = volume_opt['sea_ice_density']
    snow_density_param = volume_opt['snow_density']
    interp_missing_sit = volume_opt.get('interp_missing_sit', True)
    sic_interp_threshold = volume_opt.get('sic_interp_threshold', 15)
    resolution = config['volume']['resolution']

    sic_products = SeaIceConcentrationProducts.load_products(
        config['options']['ice_conc_products'],
        config['auxiliary']['ice_conc'],
        hem=hem,
        out_epsg=out_epsg)

    df_vol_mass = pd.DataFrame()

    for file in file_list:
        time_str = re.search(r'-(\d{8})-', os.path.basename(file)).group(1)
        dt1d = datetime.timedelta(days=1)
        t0 = datetime.datetime.strptime(time_str, '%Y%m%d')
        t1 = t0 + dt1d

        data = xr.open_dataset(file, decode_times=False)

        if which_ice_conc != None:
            sic_product = select_product(sic_products, t0, t1)
            if not sic_product:
                logger.warning(f"{t0:%Y%m%d}: no nearby concentration file; skipping")
                data.close()
                continue
            ice_conc = sic_product.get_ice_concentration(sic_product.target_files)
        else:
            ice_conc = data['sea_ice_concentration']

        dataset = calculate_volume_and_mass(
            data,
            ice_conc,
            target_var,
            si_density_param,
            snow_density_param,
            resolution,
            interp_missing_sit=interp_missing_sit,
            sic_threshold=sic_interp_threshold
        )

        outfile = out_dir + volume_opt['sub_dir'] + os.sep + os.path.basename(file)
        outfile = outfile.replace('SOSIMBA', 'SOSIMBA_VOL')
        write_volume_dataset(dataset, outfile)
        df_vol_mass_temp = build_regional_summary(dataset)
        df_vol_mass = pd.concat([df_vol_mass, df_vol_mass_temp], axis=0)
    
    df_vol_mass['time'] = pd.to_datetime(df_vol_mass.index, unit='s', origin='unix')
    df_vol_mass.set_index('time', inplace=True)

    outfile_csv = outfile.replace('.nc', '.csv').replace(outfile[-17:-9], 
                                                         '%s-%s' %(df_vol_mass.index.strftime('%Y%m%d')[0], 
                                                                  df_vol_mass.index.strftime('%Y%m%d')[-1]))
    

    organize_files_by_date(config['output_dir']['volume']+ '/' + volume_opt['sub_dir'],
                           os.path.dirname(config['output_dir']['volume']+ '/' + volume_opt['sub_dir'] + '/')) ## check 
    df_vol_mass.to_csv(outfile_csv)
    logger.info('csv files for volume saved as : %s' %outfile_csv )
