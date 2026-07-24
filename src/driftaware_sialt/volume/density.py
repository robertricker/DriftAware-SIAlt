"""Seasonal snow and sea-ice density parameterizations."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def interpolate_seasonal_density(data_seasons, time):
    dates = pd.to_datetime(time)
    start_year = dates.year - 1
    end_year = dates.year + 1
    season_dates = []
    for year in range(start_year, end_year + 1):
        for season in data_seasons["date"]:
            season_date = pd.to_datetime(
                f"{year}-{season}", format="%Y-%d-%m")
            season_dates.append(season_date)

    season_series = pd.Series(
        data_seasons["density"] * (end_year - start_year + 1),
        index=season_dates,
    )
    season_series = season_series[
        ~season_series.index.duplicated(keep="first")
    ].sort_index()
    return np.interp(
        dates.value, season_series.index, season_series.values)


def snow_density_fons_2022(time_in_seconds):
    origin = datetime(1970, 1, 1)
    time = origin + timedelta(seconds=time_in_seconds)
    data_seasons = {
        "date": ["15-07", "15-10", "15-01", "15-04", "15-07"],
        "density": [330, 310, 360, 350, 330],
    }
    return interpolate_seasonal_density(data_seasons, time)


def sea_ice_density_fons_2022(time_in_seconds):
    origin = datetime(1970, 1, 1)
    time = origin + timedelta(seconds=time_in_seconds)
    data_seasons = {
        "date": ["15-07", "15-10", "15-01", "15-04", "15-07"],
        "density": [920, 915, 875, 900, 920],
    }
    return interpolate_seasonal_density(data_seasons, time)
