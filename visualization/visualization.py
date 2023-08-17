from visualization import visualization_tools
from data_handler.sea_ice_concentration_products import SeaIceConcentrationProducts
import xarray as xr
import datetime
import matplotlib.pyplot as plt
import glob
import re
import os
from loguru import logger


def visualization(config):
    sensor = config['options']['sensor']
    hem = config["options"]["hemisphere"]
    out_epsg = config["options"]["out_epsg"]
    visu_opt = config['options']['proc_step_options']['visualization']
    target_var = visu_opt['variable']
    make_gif = visu_opt['make_gif']
    config['dir'][sensor]['netcdf'] = config['dir'][sensor]['netcdf'] + visu_opt['sub_dir']
    file_list = sorted(glob.glob(config['dir'][sensor]['netcdf'] + "/" + '*.nc'))

    out_dir = config['dir'][sensor]['visu']

    sic_product = SeaIceConcentrationProducts(hem=hem, product_id=config['options']['ice_conc_product'],
                                              out_epsg=out_epsg)
    sic_product.get_file_list(config['dir']['auxiliary']['ice_conc'][config['options']['ice_conc_product']])
    sic_product.get_file_dates()

    for file in file_list:
        time_str = re.search('nh-(.+?)-(.*).nc', os.path.basename(file)).group(1)
        dt1d = datetime.timedelta(days=1)
        t0 = datetime.datetime.strptime(time_str, '%Y%m%d')
        t1 = t0 + dt1d

        sic_product.target_files = sic_product.get_target_files(t0, t1)
        ice_conc = sic_product.get_ice_concentration(sic_product.target_files)

        data = xr.open_dataset(file, decode_times=False)

        if target_var in ['sea_ice_thickness', 'sea_ice_thickness_corrected']:
            vmin, vmax, n_level = 0, 5, 11
            cmap = plt.cm.plasma
            scaling = 1.0
            label = 'Sea ice thickness in m'

        elif target_var in ['sea_ice_thickness_unc', 'sea_ice_thickness_total_unc', 'sea_ice_thickness_drift_unc']:
            vmin, vmax, n_level = 0, 0.6, 13
            cmap = plt.cm.plasma
            scaling = 1.0
            label = 'Sea ice thickness uncertainty in m'

        elif target_var in ['sea_ice_freeboard', 'sea_ice_freeboard_corrected']:
            vmin, vmax, n_level = 0, 60, 13
            cmap = plt.cm.plasma
            scaling = 100.0
            label = 'Sea ice freeboard in cm'

        elif target_var in ['sea_ice_freeboard_unc', 'sea_ice_freeboard_total_unc', 'sea_ice_freeboard_drift_unc']:
            vmin, vmax, n_level = 0, 6, 13
            cmap = plt.cm.plasma
            scaling = 100.0
            label = 'Sea ice freeboard uncertainty in cm'

        elif target_var in ['total_freeboard', 'total_freeboard_corrected']:
            vmin, vmax, n_level = 0, 60, 13
            cmap = plt.cm.plasma
            scaling = 100.0
            label = 'Total freeboard in cm'

        elif target_var in ['total_freeboard_unc', 'total_freeboard_total_unc', 'total_freeboard_drift_unc']:
            vmin, vmax, n_level = 0, 6, 13
            cmap = plt.cm.plasma
            scaling = 100.0
            label = 'Total freeboard uncertainty in cm'

        elif target_var == "dist_acquisition":
            vmin, vmax, n_level = 0, 160, 17
            cmap = plt.cm.plasma
            scaling = 1.0
            label = 'Distance to data aquisition in km'

        elif target_var == "time_offset_acquisition":
            vmin, vmax, n_level = -15, 15, 13
            cmap = plt.cm.bwr
            scaling = 1.0
            label = 'time offfset to data aquisition in days'

        elif target_var == "sea_ice_freeboard_growth_interpolated":
            vmin, vmax, n_level = -0.5, 0.5, 20
            cmap = plt.cm.bwr
            scaling = 100.0
            label = 'Sea ice freeboard growth in cm/day'

        elif target_var == "sea_ice_thickness_growth_interpolated":
            vmin, vmax, n_level = -5, 5, 20
            cmap = plt.cm.bwr
            scaling = 100.0
            label = 'Total freeboard growth in cm/day'

        else:
            break

        if not os.path.exists(out_dir + target_var):
            try:
                os.mkdir(out_dir + target_var)
            except OSError as error:
                print(error)

        outfile = out_dir + target_var + os.sep + re.split('.nc', os.path.basename(file))[0] + '_' + target_var + '.png'
        visualization_tools.visu_xarray(data.xc, data.yc, data[target_var] * scaling,
                                        (8, 8),
                                        vmin, vmax, n_level,
                                        cmap,
                                        time_str,
                                        label,
                                        outfile,
                                        iceconc=ice_conc)

    if make_gif:
        visualization_tools.make_gif(out_dir+target_var, target_var)
