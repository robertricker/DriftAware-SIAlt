from visualization import visualization_tools
from data_handler.sea_ice_concentration_products import SeaIceConcentrationProducts
import xarray as xr
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob
import re
import os
from loguru import logger


def create_diverging_colormap():
    colors = [
        (0.4, 0, 0.4),
        (1.00, 1.00, 1.00),
        (1.00, 0.4, 0.05)]
    cmap = LinearSegmentedColormap.from_list('PuOr', colors, N=256)
    return cmap


def visualization(config):
    sensor = config['options']['sensor']
    hem = config["options"]["hemisphere"]
    out_epsg = config["options"]["out_epsg"]
    visu_opt = config['options']['proc_step_options']['visualization']
    target_var = visu_opt['variable']
    make_gif = visu_opt['make_gif']
    config['dir'][sensor]['netcdf'] = config['dir'][sensor]['netcdf'] + visu_opt['sub_dir']
    file_list = sorted(glob.glob(os.path.join(config['dir'][sensor]['netcdf'], '**', '*.nc'), recursive=True))
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
            cmap = plt.cm.cool
            scaling = 1.0
            label = 'Sea ice thickness (m)'

        elif target_var in ['sea_ice_thickness_l2_unc', 'sea_ice_thickness_total_unc',
                            'sea_ice_thickness_growth_unc', 'sea_ice_thickness_drift_unc']:
            vmin, vmax, n_level = 0, 1.0, 13
            cmap = plt.cm.cool
            scaling = 1.0
            label = 'Sea ice thickness uncertainty (m)'

        elif target_var in ['sea_ice_freeboard', 'sea_ice_freeboard_uncorrected']:
            vmin, vmax, n_level = 0, 60, 13
            cmap = plt.cm.cool
            scaling = 100.0
            label = 'Sea ice freeboard (cm)'

        elif target_var in ['sea_ice_freeboard_l2_unc', 'sea_ice_freeboard_total_unc', 'sea_ice_freeboard_drift_unc']:
            vmin, vmax, n_level = 0, 6, 13
            cmap = plt.cm.cool
            scaling = 100.0
            label = 'Sea ice freeboard uncertainty (cm)'

        elif target_var in ['total_freeboard', 'total_freeboard_corrected']:
            vmin, vmax, n_level = 0, 60, 13
            cmap = plt.cm.cool
            scaling = 100.0
            label = 'Total freeboard in cm'

        elif target_var in ['total_freeboard_l2_unc', 'total_freeboard_total_unc', 'total_freeboard_drift_unc']:
            vmin, vmax, n_level = 0, 6, 13
            cmap = plt.cm.cool
            scaling = 100.0
            label = 'Total freeboard uncertainty (cm)'

        elif target_var == "dist_acquisition":
            vmin, vmax, n_level = 0, 160, 17
            cmap = plt.cm.cool
            scaling = 1.0
            label = 'Distance to data acquisition (km)'

        elif target_var == "time_offset_acquisition":
            vmin, vmax, n_level = -15, 15, 13
            cmap = create_diverging_colormap()
            scaling = 1.0
            label = 'Time offset to data acquisition (days)'

        elif target_var == "sea_ice_freeboard_growth_interpolated":
            vmin, vmax, n_level = -0.5, 0.5, 20
            cmap = create_diverging_colormap()
            scaling = 100.0
            label = 'Sea ice freeboard growth (cm day$^{-1}$)'

        elif target_var in ['sea_ice_thickness_growth', 'sea_ice_thickness_growth_interpolated']:
            vmin, vmax, n_level = -5, 5, 20
            cmap = create_diverging_colormap()
            scaling = 100.0
            label = 'Sea ice thickness growth (cm day$^{-1}$)'

        elif target_var in ["shear", "divergence"]:
            vmin, vmax, n_level = -0.1, 0.1, 20
            cmap = create_diverging_colormap()
            scaling = -1.0
            label = 'Convergence (day$^{-1}$)'

        else:
            break

        if not os.path.exists(out_dir + target_var):
            try:
                os.mkdir(out_dir + target_var)
            except OSError as error:
                print(error)

        outfile = out_dir + target_var + os.sep + re.split('.nc', os.path.basename(file))[0] + '_' + target_var + '.png'
        print(outfile)
        visualization_tools.visu_xarray(data.xc, data.yc, data[target_var][0] * scaling,
                                        (6, 6),
                                        vmin, vmax, n_level,
                                        cmap,
                                        time_str,
                                        label,
                                        outfile,
                                        iceconc=ice_conc)

    if make_gif:
        visualization_tools.make_gif(out_dir+target_var, target_var)
