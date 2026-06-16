import numpy as np 
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import RBFInterpolator
from data_handler.sea_ice_concentration_products import SeaIceConcentrationProducts

#b volume/compute_volume.py:13

def interp_season(data_seasons, time):
    dates = pd.to_datetime(time)
    start_year = dates.year - 1
    end_year = dates.year + 1
    season_dates = []
    for year in range(start_year, end_year + 1):
        for season in data_seasons["date"]:
            season_date = pd.to_datetime(f"{year}-{season}", format="%Y-%d-%m")
            season_dates.append(season_date)
        
    season_series = pd.Series(data_seasons["density"] * (end_year - start_year + 1), index=season_dates)
    season_series = season_series[~season_series.index.duplicated(keep='first')].sort_index()
    dens = np.interp(dates.value, season_series.index, season_series.values)
    return dens

def snow_density_fons(time_in_seconds):

    origin = datetime(1970, 1, 1)
    time = origin + timedelta(seconds=time_in_seconds)
    data_seasons = {
         "date": ["15-07", "15-10", "15-01", "15-04", "15-07"],
         "density": [330, 310, 360, 350, 330],
         }
    dens = interp_season(data_seasons, time)
    return dens

def sea_ice_density_fons(time_in_seconds):

    origin = datetime(1970, 1, 1)
    time = origin + timedelta(seconds=time_in_seconds)
    data_seasons = {
         "date": ["15-07", "15-10", "15-01", "15-04", "15-07"],
         "density": [920, 915, 875, 900, 920],
         }
    dens = interp_season(data_seasons, time)
    return dens


def interpolate_target_var_rbf(data, target_var, ice_conc, sic_threshold=0.15):
    """Interpolate missing target variable using RBF where ice concentration exceeds threshold."""
    if target_var not in data:
        return data

    # Normalize ice concentration array
    if isinstance(ice_conc, dict):
        ice_conc_arr = ice_conc.get('ice_conc')
    else:
        ice_conc_arr = ice_conc.values if hasattr(ice_conc, 'values') else np.array(ice_conc)

    if ice_conc_arr.ndim == 3 and ice_conc_arr.shape[0] == 1:
        ice_conc_arr = ice_conc_arr[0]

    da = data[target_var]
    
    # Handle time dimension
    if 'time' in da.dims and da.sizes.get('time', 1) > 1:
        # Process each time step
        for t in range(da.sizes['time']):
            da_t = da.isel(time=t)
            if 'xc' in da_t.dims and 'yc' in da_t.dims:
                data[target_var].values[t] = _interpolate_2d_rbf(
                    da_t.values, ice_conc_arr, da_t['xc'].values, da_t['yc'].values, sic_threshold
                )
    elif 'xc' in da.dims and 'yc' in da.dims:
        if da.sizes.get('time') == 1:
            da_vals = da.values[0]
        else:
            da_vals = da.values

        if da_vals.ndim == 3 and da_vals.shape[0] == 1:
            da_vals = da_vals[0]

        filled_vals = _interpolate_2d_rbf(
            da_vals, ice_conc_arr, da['xc'].values, da['yc'].values, sic_threshold
        )

        if 'time' in da.dims and da.sizes.get('time', 1) == 1:
            data[target_var].values[0] = filled_vals
        else:
            data[target_var].values = filled_vals

    return data


def _interpolate_2d_rbf(da_vals, ice_conc_arr, xc, yc, sic_threshold):
    """RBF Gaussian interpolation for 2D array."""
    #valid = ~np.isnan(da_vals)
    #mask = np.isnan(da_vals) & (ice_conc_arr > sic_threshold)
    import matplotlib.pyplot as plt
    #if not np.any(mask) or np.sum(valid) < 4:
    #    return da_vals.copy()

    X, Y = np.meshgrid(xc, yc)
    mask_valid = ~np.isnan(da_vals.squeeze())
    x_known = X[mask_valid]
    y_known = Y[mask_valid]
    points_known = np.column_stack((x_known, y_known))
    z_known = da_vals.squeeze()[mask_valid]
    
    try:
        rbf = RBFInterpolator(points_known, z_known, kernel='gaussian', epsilon=1.8/25000, neighbors=20, smoothing=0.05)
        mask_interp = ice_conc_arr.squeeze() > 0.15
        x_all = X[mask_interp]
        y_all = Y[mask_interp]
        points_all = np.column_stack((x_all, y_all))
        z_all = rbf(points_all)
        
        data_filled = np.full_like(da_vals.squeeze(), np.nan)
        data_filled[mask_interp] = z_all
        plt.imshow(data_filled)
        plt.savefig("test.png")
        filled = np.expand_dims(data_filled, axis=0)
        return filled
    
    except Exception as e:
        print(f"RBF interpolation failed: {e}. Returning original data.")
        return da_vals.copy()

def compute_volume_and_mass(data, ice_conc, target_var, si_density_param, snow_density_param, resolution,
                            interp_missing_sit=True, sic_threshold=0.15):
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
        'snow_fons_2022': snow_density_fons,
        'ice_fons_2022': sea_ice_density_fons,
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
        data = interpolate_target_var_rbf(data, target_var, ice_conc_arr, sic_threshold=sic_threshold)
    if interp_missing_sit and 'snow_depth' in data:
        data = interpolate_target_var_rbf(data, 'snow_depth', ice_conc_arr, sic_threshold=sic_threshold)

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

