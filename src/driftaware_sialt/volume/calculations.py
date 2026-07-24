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


def calculate_volume_and_mass(
        data, ice_conc, target_var, si_density_param, snow_density_param,
        resolution, interp_missing_sit=True, sic_threshold=0.15):
    # Normalize ice concentration and align it to the data grid if needed
    if isinstance(ice_conc, dict):
        xc_target = data['xc'].values
        yc_target = data['yc'].values
        Xt, Yt = np.meshgrid(xc_target, yc_target)
        pts = np.column_stack((Xt.ravel(), Yt.ravel()))
        ice_conc_arr = SeaIceConcentrationProducts.interp_ice_concentration(ice_conc, pts[:, 0], pts[:, 1]).reshape(Xt.shape)
    else:
        ice_conc_arr = ice_conc

    if np.nanmean(np.asarray(ice_conc_arr)) > 1:
        ice_conc_arr *= 0.01
    
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
    cell_area = (int(resolution) * 1000) ** 2

    ice_conc_da = xr.DataArray(
        ice_conc_arr,
        coords={
            'yc': data['yc'].values,
            'xc': data['xc'].values,
        },
        dims=('yc', 'xc')
    )

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
