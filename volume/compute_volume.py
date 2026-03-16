import numpy as np 
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
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

def compute_volume_and_mass(data, ice_conc, target_var, si_density_param, snow_density_param, resolution):
    if ice_conc.mean()>1:
        ice_conc*= 0.01
    
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
    

    data["_".join(target_var.rsplit("_", 1)[:-1] + ["volume"])] = data[target_var] * ice_conc * (int(resolution) * 1000)**2
    data["_".join(target_var.rsplit("_", 1)[:-1] + ["mass"])] = data["_".join(target_var.rsplit("_", 1)[:-1] + ["volume"])] * ice_density

    data['snow_volume'] = data['snow_depth'] * ice_conc * (int(resolution) * 1000)**2
    data['snow_mass'] = data['snow_volume'] * snow_density
    return data



