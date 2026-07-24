from driftaware_sialt.visualization import visualization_tools
import xarray as xr
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
    visu_opt = config['visualization']
    target_var = visu_opt['variable']
    make_gif = visu_opt['make_gif']
    config['output_dir']['gridded_data'] = os.path.join(
        config['output_dir']['gridded_data'], visu_opt['sub_dir'])
    file_list = sorted(glob.glob(os.path.join(config['output_dir']['gridded_data'], '**', '*.nc'), recursive=True))
    out_dir = config['output_dir']['visu']

    for file in file_list:
        time_str = re.search(r'(\d{8})', os.path.basename(file)).group(1)

        data = xr.open_dataset(file, decode_times=False)

        if target_var in ['sea_ice_thickness', 'sea_ice_thickness_corrected']:
            vmin, vmax, n_level = 0, 5, 11
            cmap = plt.cm.cool
            scaling = 1.0
            label = 'Sea ice thickness (m)'

        elif target_var in ['sea_ice_thickness_l2_unc', 'sea_ice_thickness_total_unc',
                            'sea_ice_thickness_change_unc', 'sea_ice_thickness_drift_unc']:
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
        elif target_var in ['snow_depth', 'snow_depth_uncorrected']:
            vmin, vmax, n_level = 0, 50, 13
            cmap = plt.cm.magma
            scaling = 100.0
            label = 'Snow Depth in cm'

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

        elif target_var == "sea_ice_freeboard_change_interpolated":
            vmin, vmax, n_level = -0.5, 0.5, 20
            cmap = create_diverging_colormap()
            scaling = 100.0
            label = 'Sea ice freeboard change (cm day$^{-1}$)'

        elif target_var in ['sea_ice_thickness_change', 'sea_ice_thickness_change_interpolated']:
            vmin, vmax, n_level = -5, 5, 20
            cmap = create_diverging_colormap()
            scaling = 100.0
            label = 'Sea ice thickness change (cm day$^{-1}$)'

        elif target_var in ["shear", "divergence"]:
            vmin, vmax, n_level = -0.1, 0.1, 20
            cmap = create_diverging_colormap()
            scaling = -1.0
            label = 'Convergence (day$^{-1}$)'

        else:
            break

        target_dir = os.path.join(out_dir, target_var)
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir, exist_ok=True)
            except OSError as error:
                print(error)

        outfile = os.path.join(
            target_dir,
            re.split('.nc', os.path.basename(file))[0] + '_' + target_var + '.png')
        print(outfile)
        if not os.path.exists(os.path.dirname(outfile)):
            try:
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
            except OSError as error:
                print(error)
        visualization_tools.visu_xarray(data.xc, data.yc, data[target_var][0] * scaling,
                                        (6, 6),
                                        vmin, vmax, n_level,
                                        cmap,
                                        time_str,
                                        label,
                                        outfile,
                                        hem)
                                        #,iceconc=ice_conc)

    if make_gif:
        visualization_tools.make_gif(os.path.join(out_dir, target_var), target_var)
