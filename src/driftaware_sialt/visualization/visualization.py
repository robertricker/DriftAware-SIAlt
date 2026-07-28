"""Render gridded NetCDF variables using consistent plotting presets."""

from dataclasses import dataclass
import glob
import os
import re

import cmocean
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap
import xarray as xr

from driftaware_sialt.visualization import visualization_tools


@dataclass(frozen=True)
class PlotPreset:
    vmin: float
    vmax: float
    levels: int
    cmap: Colormap
    scale: float
    label: str


def create_diverging_colormap():
    colors = [
        (0.4, 0, 0.4),
        (1.0, 1.0, 1.0),
        (1.0, 0.4, 0.05),
    ]
    return LinearSegmentedColormap.from_list('PuOr', colors, N=256)


_DIVERGING_CMAP = create_diverging_colormap()

PLOT_PRESETS = {
    'sea_ice_thickness': PlotPreset(
        0, 5, 21, cmocean.cm.thermal, 1.0, 'Sea ice thickness (m)'),
    'sea_ice_thickness_uncorrected': PlotPreset(
        0, 5, 21, cmocean.cm.thermal, 1.0,
        'Uncorrected sea ice thickness (m)'),
    'sea_ice_thickness_mode': PlotPreset(
        0, 5, 21, cmocean.cm.thermal, 1.0,
        'Modal sea ice thickness (m)'),
    'model_sea_ice_thickness': PlotPreset(
        0, 5, 21, cmocean.cm.thermal, 1.0,
        'Model sea ice thickness (m)'),
    'sea_ice_thickness_std': PlotPreset(
        0, 1, 13, plt.cm.cool, 1.0,
        'Sea ice thickness standard deviation (m)'),
    'sea_ice_thickness_l2_unc': PlotPreset(
        0, 1, 13, plt.cm.cool, 1.0,
        'Level-2 sea ice thickness uncertainty (m)'),
    'sea_ice_thickness_drift_unc': PlotPreset(
        0, 1, 13, plt.cm.cool, 1.0,
        'Drift-related sea ice thickness uncertainty (m)'),
    'sea_ice_thickness_change_unc': PlotPreset(
        0, 1, 13, plt.cm.cool, 1.0,
        'Change-related sea ice thickness uncertainty (m)'),
    'sea_ice_thickness_total_unc': PlotPreset(
        0, 1, 13, plt.cm.cool, 1.0,
        'Total sea ice thickness uncertainty (m)'),
    'radar_freeboard': PlotPreset(
        0, 60, 13, plt.cm.cool, 100.0, 'Radar freeboard (cm)'),
    'sea_ice_freeboard': PlotPreset(
        0, 60, 13, plt.cm.cool, 100.0, 'Sea ice freeboard (cm)'),
    'snow_depth': PlotPreset(
        0, 50, 13, plt.cm.magma, 100.0, 'Snow depth (cm)'),
    'sea_ice_concentration': PlotPreset(
        0, 100, 21, cmocean.cm.ice, 1.0, 'Sea ice concentration (%)'),
    'dist_acquisition': PlotPreset(
        0, 160, 17, plt.cm.cool, 1.0,
        'Distance to data acquisition (km)'),
    'time_offset_acquisition': PlotPreset(
        -15, 15, 13, _DIVERGING_CMAP, 1.0,
        'Time offset to data acquisition (days)'),
    'sea_ice_thickness_change': PlotPreset(
        -5, 5, 20, _DIVERGING_CMAP, 100.0,
        'Sea ice thickness change (cm day$^{-1}$)'),
    'sea_ice_thickness_change_interpolated': PlotPreset(
        -5, 5, 20, _DIVERGING_CMAP, 100.0,
        'Interpolated sea ice thickness change (cm day$^{-1}$)'),
    'dynamic_change_rate': PlotPreset(
        -5, 5, 20, _DIVERGING_CMAP, 100.0,
        'Dynamic sea ice thickness change (cm day$^{-1}$)'),
    'sea_ice_thickness_thermodynamic_tendency': PlotPreset(
        -5, 5, 20, _DIVERGING_CMAP, 100.0,
        'Sea ice thickness thermodynamic tendency (cm day$^{-1}$)'),
    'model_sea_ice_thickness_change_rate': PlotPreset(
        -5, 5, 20, _DIVERGING_CMAP, 100.0,
        'Model sea ice thickness change (cm day$^{-1}$)'),
    'model_sea_ice_thickness_thermodynamic_tendency': PlotPreset(
        -5, 5, 20, _DIVERGING_CMAP, 100.0,
        'Model thermodynamic tendency (cm day$^{-1}$)'),
    'deformation': PlotPreset(
        0, 0.1, 21, cmocean.cm.amp, 1.0, 'Deformation (day$^{-1}$)'),
    'shear': PlotPreset(
        0, 0.1, 21, cmocean.cm.amp, 1.0, 'Shear (day$^{-1}$)'),
    'divergence': PlotPreset(
        -0.1, 0.1, 21, _DIVERGING_CMAP, 1.0,
        'Divergence (day$^{-1}$)'),
    'sea_ice_displacement_uncertainty': PlotPreset(
        0, 160, 17, plt.cm.cool, 0.001,
        'Sea ice displacement uncertainty (km)'),
    'model_air_temperature': PlotPreset(
        -40, 5, 19, cmocean.cm.thermal, 1.0,
        'Model air temperature (°C)'),
    'model_ocean_heat_flux': PlotPreset(
        0, 100, 21, cmocean.cm.thermal, 1.0,
        'Model ocean heat flux (W m$^{-2}$)'),
}


def _find_input_files(gridded_data_dir, sub_dir):
    input_dir = (
        sub_dir
        if os.path.isabs(sub_dir)
        else os.path.join(gridded_data_dir, sub_dir)
    )
    pattern = os.path.join(input_dir, '**', '*.nc')
    return input_dir, sorted(glob.glob(pattern, recursive=True))


def _extract_date(file):
    match = re.search(r'(\d{8})', os.path.basename(file))
    return match.group(1) if match else None


def _first_time_slice(variable):
    if 'time' in variable.dims:
        variable = variable.isel(time=0)
    if variable.ndim != 2:
        raise ValueError(
            f"expected a two-dimensional map, got dimensions {variable.dims}")
    return variable


def visualization(config):
    hem = config['options']['hemisphere']
    options = config['visualization']
    target_var = options['variable']

    try:
        preset = PLOT_PRESETS[target_var]
    except KeyError as error:
        supported = ', '.join(sorted(PLOT_PRESETS))
        raise ValueError(
            f"no visualization preset for '{target_var}'; "
            f"supported variables: {supported}") from error

    input_dir, file_list = _find_input_files(
        config['output_dir']['gridded_data'], options['sub_dir'])
    if not file_list:
        logger.warning('No NetCDF files found for visualization in: {}',
                       input_dir)
        return

    target_dir = os.path.join(config['output_dir']['visu'], target_var)
    os.makedirs(target_dir, exist_ok=True)
    rendered_files = 0

    for file in file_list:
        time_string = _extract_date(file)
        if time_string is None:
            logger.warning('Skipping file without an eight-digit date: {}', file)
            continue

        with xr.open_dataset(file, decode_times=False) as data:
            if target_var not in data:
                logger.warning("Skipping {}: variable '{}' is missing",
                               file, target_var)
                continue

            values = _first_time_slice(data[target_var]) * preset.scale
            outfile = os.path.join(
                target_dir,
                f"{os.path.splitext(os.path.basename(file))[0]}"
                f"_{target_var}.png",
            )
            logger.info('Writing visualization: {}', outfile)
            visualization_tools.visu_xarray(
                data.xc,
                data.yc,
                values,
                (6, 6),
                preset.vmin,
                preset.vmax,
                preset.levels,
                preset.cmap,
                time_string,
                preset.label,
                outfile,
                hem,
            )
            rendered_files += 1

    if options['make_gif'] and rendered_files:
        visualization_tools.make_gif(target_dir, target_var)
