"""Calculations of gridded sea-ice and snow bulk properties."""

import numpy as np
import xarray as xr

from driftaware_sialt.products.sea_ice_concentration import (
    SeaIceConcentrationProducts,
)
from driftaware_sialt.volume.density import (
    sea_ice_density_fons_2022,
    snow_density_fons_2022,
)
from driftaware_sialt.volume.interpolation import interpolate_missing_values


def infer_grid_cell_area(data):
    """Return uniform grid-cell area in square metres from x/y coordinates."""
    dx = np.abs(np.diff(data['xc'].values))
    dy = np.abs(np.diff(data['yc'].values))
    if (
        not dx.size or not dy.size
        or dx[0] == 0 or dy[0] == 0
        or not np.allclose(dx, dx[0])
        or not np.allclose(dy, dy[0])
    ):
        raise ValueError('xc and yc must have regular, non-zero spacing')
    return float(dx[0] * dy[0])


def calculate_volume_and_mass(
        data, ice_conc, target_var, si_density_param, snow_density_param,
        interp_missing_sit=True, sic_threshold=0.15):
    if target_var != 'sea_ice_thickness':
        raise ValueError(
            'volume calculation requires target_variable: '
            'sea_ice_thickness')

    # Normalize ice concentration and align it to the data grid if needed
    if isinstance(ice_conc, dict):
        xc_target = data['xc'].values
        yc_target = data['yc'].values
        Xt, Yt = np.meshgrid(xc_target, yc_target)
        pts = np.column_stack((Xt.ravel(), Yt.ravel()))
        ice_conc_arr = SeaIceConcentrationProducts.interp_ice_concentration(ice_conc, pts[:, 0], pts[:, 1]).reshape(Xt.shape)
    else:
        ice_conc_arr = np.asarray(ice_conc).squeeze()

    if np.nanmean(np.asarray(ice_conc_arr)) > 1:
        ice_conc_arr *= 0.01
    if np.ndim(ice_conc_arr) != 2:
        raise ValueError(
            'sea-ice concentration must resolve to a two-dimensional grid')
    if sic_threshold > 1:
        sic_threshold *= 0.01
    
    f_density = {
        'snow_fons_2022': snow_density_fons_2022,
        'ice_fons_2022': sea_ice_density_fons_2022,
        }
    
    if type(si_density_param) == str:
        ice_density = f_density[si_density_param](data.time.data[0])
    else:
        ice_density = si_density_param
    
    if type(snow_density_param) == str:
        snow_density = f_density[snow_density_param](data.time.data[0])
    else:
        snow_density = snow_density_param
    
    # Interpolate missing target variable if requested
    if interp_missing_sit and target_var in data:
        data = interpolate_missing_values(
            data, target_var, ice_conc_arr,
            sic_threshold=sic_threshold)
    if interp_missing_sit and 'snow_depth' in data:
        data = interpolate_missing_values(
            data, 'snow_depth', ice_conc_arr,
            sic_threshold=sic_threshold)

    sea_ice_volume_name = "_".join(target_var.rsplit("_", 1)[:-1] + ["volume"])
    sea_ice_mass_name = "_".join(target_var.rsplit("_", 1)[:-1] + ["mass"])
    cell_area = infer_grid_cell_area(data)

    ice_conc_da = xr.DataArray(
        ice_conc_arr,
        coords={
            'yc': data['yc'].values,
            'xc': data['xc'].values,
        },
        dims=('yc', 'xc')
    )
    if 'time' in data[target_var].dims:
        ice_conc_da = ice_conc_da.expand_dims(time=data['time'])

    data[sea_ice_volume_name] = data[target_var] * ice_conc_da * cell_area
    data[sea_ice_mass_name] = data[sea_ice_volume_name] * ice_density

    data['snow_volume'] = data['snow_depth'] * ice_conc_da * cell_area
    data['snow_mass'] = data['snow_volume'] * snow_density

    data['sea_ice_extent'] = (ice_conc_da > sic_threshold).astype(np.float32) * cell_area
    data['sea_ice_area'] = ice_conc_da * cell_area

    data['sea_ice_extent_total'] = data['sea_ice_extent'].sum(dim=['xc', 'yc'])
    data['sea_ice_area_total'] = data['sea_ice_area'].sum(dim=['xc', 'yc'])
    data['sea_ice_volume_total'] = data[sea_ice_volume_name].sum(dim=['xc', 'yc'])
    data['sea_ice_mass_total'] = data[sea_ice_mass_name].sum(dim=['xc', 'yc'])

    return data
