from data_handler.sea_ice_concentration_products import SeaIceConcentrationProducts
import xarray as xr
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from gridding.prepare_netcdf import PrepareNetcdf
from volume import compute_volume
import pandas as pd
import yaml
import shutil
import glob
import re
import os
from loguru import logger
from jinja2 import Template

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


def volume(config):
    sensor = config['options']['sensor']
    hem = config["options"]["hemisphere"]
    out_epsg = config["options"]["out_epsg"]
    volume_opt = config['options']['proc_step_options']['volume']
    which_ice_conc = volume_opt['ice_conc']
    target_var = config['options']['target_variable']

    config['output_dir']['volume_data'] = config['output_dir']['volume'] + '/' + volume_opt['sub_dir']
    file_list = sorted(glob.glob(os.path.join(config['output_dir']['gridded_data']+volume_opt['sub_dir'], '**', '*.nc'), recursive=True))
    out_dir = config['output_dir']['volume']
    si_density_param = volume_opt['sea_ice_density']
    snow_density_param = volume_opt['snow_density']
    mode = file_list[0][-20:-18].lower()

    resolution = file_list[0][-31:-29]

    sic_product = SeaIceConcentrationProducts(hem=hem, product_id=config['options']['ice_conc_product'],
                                              out_epsg=out_epsg)
    sic_product.get_file_list(config['auxiliary']['ice_conc'][config['options']['ice_conc_product']])
    sic_product.get_file_dates()

    df_vol_mass = pd.DataFrame()

    for file in file_list:
        time_str = re.search(r'-(\d{8})-', os.path.basename(file)).group(1)
        dt1d = datetime.timedelta(days=1)
        t0 = datetime.datetime.strptime(time_str, '%Y%m%d')
        t1 = t0 + dt1d

        data = xr.open_dataset(file, decode_times=False)

        if which_ice_conc != None:
            sic_product.target_files = sic_product.get_target_files(t0, t1)
            ice_conc = sic_product.get_ice_concentration(sic_product.target_files)
        else:
            ice_conc = data['sea_ice_concentration']

        dataset = compute_volume.compute_volume_and_mass(data, ice_conc, target_var, si_density_param, snow_density_param, resolution)

        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except OSError as error:
                print(error)

        outfile = out_dir + volume_opt['sub_dir'] + os.sep + os.path.basename(file)
        outfile = outfile.replace('SOSIMBA', 'SOSIMBA_VOL')
        
        with open(os.path.join(os.path.dirname(__file__), 'netcdf_config.yaml'), 'r') as f:
            netcdf_config = yaml.safe_load(f)
        

        ## Create the netcdf with volumes
        var = [Template(item).render(target_var=target_var) #Exception for is2
               for item in netcdf_config['variables'][mode]['include']
               if not (sensor == 'icesat2' and item == 'snow_depth')] 
        var_rename = [Template(item).render(target_var=target_var) #Same exception for is2
                      for item in netcdf_config['variables'][mode]['rename']
                      if not (sensor == 'icesat2' and item == 'snow_depth')] 
        


        variable_attributes = netcdf_config['variable_attributes']
        for var_name, attributes in variable_attributes.items():
            if var_name in dataset:
                dataset[var_name].attrs.update(attributes)
        
        
        if not os.path.exists(os.path.dirname(outfile)):
            try:
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
            except OSError as error:
                print(error)
        comp = dict(zlib=True, complevel=1)
        encoding = {var: comp for var in dataset.data_vars}
        for var in dataset.variables:
            dataset[var].encoding = encoding.copy()  # Apply the new encoding for each variable

        dataset.to_netcdf(outfile, encoding = encoding, format="NETCDF4")
        logger.info('Volumes and masses saved as : %s' %outfile)

        # Update the csv file with volumes and mass
        ds_vol_mass = dataset[['snow_volume', 
                                    'snow_mass', 
                                    'sea_ice_volume', 
                                    'sea_ice_mass']]
        df_vol_mass_temp = ds_vol_mass.sum(dim=['xc', 'yc']).to_dataframe()
        
        region_flag = dataset.region_flag
        region_codes = region_flag.attrs["flag_values"]
        region_names = region_flag.attrs["flag_meanings"].split()        #unique_regions = np.unique(regions[regions!=0])  # suppress the 0 (undefined region)

        for region in region_codes:

            # select data for a specific region :
            df_region = ds_vol_mass.where(dataset.region_flag == region).sum(dim=['xc', 'yc']).to_dataframe()/1000000000
            df_region_renamed = df_region.rename(columns={
                var: f"{var}_{region}" for var in df_region.columns
                })
            
            df_vol_mass_temp = pd.concat([df_vol_mass_temp, df_region_renamed], axis=1)

        

        df_vol_mass = pd.concat([df_vol_mass, df_vol_mass_temp], axis=0)
    
    df_vol_mass['time'] = pd.to_datetime(df_vol_mass.index, unit='s', origin='unix')
    df_vol_mass.set_index('time', inplace=True)

    outfile_csv = outfile.replace('.nc', '.csv').replace(outfile[-17:-9], 
                                                         '%s-%s' %(df_vol_mass.index.strftime('%Y%m%d')[0], 
                                                                  df_vol_mass.index.strftime('%Y%m%d')[-1]))
    

    organize_files_by_date(config['output_dir']['volume']+ '/' + volume_opt['sub_dir'],
                           os.path.dirname(config['output_dir']['volume']+ '/' + volume_opt['sub_dir'])) ## check 
    df_vol_mass.to_csv(outfile_csv)
    logger.info('csv files for volume saved as : %s' %outfile_csv )



        