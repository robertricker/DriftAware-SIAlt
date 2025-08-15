import numpy as np
import xarray as xr
import geopandas as gpd
from scipy import ndimage
from scipy.spatial.distance import cdist
from pyproj import Proj
from shapely.geometry import Point
import pandas as pd

def compute_projection(griddef):
    return Proj(**griddef["projection"])

def get_image_coords(lon, lat, proj, grid_lons, grid_lats, griddef):
    x_grid, y_grid = proj(grid_lons, grid_lats)
    x_min, y_min = np.nanmin(x_grid), np.nanmin(y_grid)
    x_track, y_track = proj(lon, lat)
    ix = (x_track - x_min) / griddef["dimension"]["dx"]
    iy = (y_track - y_min) / griddef["dimension"]["dy"]
    return ix, iy

def extract_along_track(gridvar, ix, iy, flipud=False, order=1):
    if flipud:
        gridvar = np.flipud(gridvar)
    return ndimage.map_coordinates(gridvar, [iy, ix], order=order)

def compute_proximity(ice_conc, threshold, pixel_spacing):
    ice_mask = ice_conc >= threshold
    ocean_mask = ice_conc < threshold
    ice_mask_ext = ndimage.maximum_filter(ice_mask, 3)

    flag = np.full_like(ice_conc, -1, dtype=int)
    flag[ice_mask] = 1
    flag[ocean_mask] = 0
    gx, gy = np.gradient(flag)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    edge = (grad > 0) & (flag == 0) & (ice_mask_ext)

    ice_idx = np.array(np.where(ice_mask)).T
    edge_idx = np.array(np.where(edge)).T
    dist = cdist(ice_idx, edge_idx)
    min_dist = np.nanmin(dist, axis=1) * pixel_spacing

    proximity = np.full_like(ice_conc, np.nan, dtype=float)
    proximity[tuple(ice_idx.T)] = min_dist
    return proximity

def add_sic_variables_to_gdf(gdf, ds_ice_conc, griddef):
    # All the incoming products should have been resampled to the same griddef (see exceptions for other resolutions ? )
    griddef = {
        "projection": {
            "proj": "laea",
            "ellps": "WGS84",
            "lon_0": 0,
            "lat_0": -90,
            "units": "m"
        },
        "dimension": {
            "n_cols": 432,
            "n_lines": 432,
            "dx": 25000,
            "dy": 25000
        }
    }

    # Load SIC grid
    ice_conc = ds_ice_conc["ice_conc"][:, :]
    grid_lons = ds_ice_conc["lon"]
    grid_lats = ds_ice_conc["lat"]

    # Mask invalid
    ice_conc = np.where(ice_conc < 0, np.nan, ice_conc)

    # Proj and track coords
    proj = compute_projection(griddef)
    ix, iy = get_image_coords(gdf["longitude"].values, gdf["latitude"].values,
                              proj, grid_lons, grid_lats, griddef)

    # Interpolate SIC
    sic_track = extract_along_track(ice_conc, ix, iy, flipud=True)

    # Compute distances
    dx = float(griddef["dimension"]["dx"]) # can be determined without the griddef
    dist_to_ocean = compute_proximity(ice_conc, 15., dx)
    dist_to_low_sic = compute_proximity(ice_conc, 70., dx)

    # Interpolate distances
    dist_ocean_track = extract_along_track(dist_to_ocean, ix, iy, flipud=True, order=1)
    dist_low_sic_track = extract_along_track(dist_to_low_sic, ix, iy, flipud=True, order=1)

    # Mask distance where SIC < 15%
    dist_ocean_track[sic_track < 15.] = np.nan

    # Add to GeoDataFrame
    gdf["sic"] = sic_track
    gdf["distance_to_ocean_km"] = dist_ocean_track / 1000
    gdf["distance_to_low_sic_km"] = dist_low_sic_track / 1000

    return gdf

def add_tFB_clim_variables_to_gdf(gdf, ds_clim_product, griddef):
    # Load clim tFB grid
    tFB_interp = ds_clim_product["tFB_interp"]
    sigma_tFB_interp = ds_clim_product["sigma_tFB_interp"]
    grid_lons = ds_clim_product["longitude"]
    grid_lats = ds_clim_product["latitude"]

    # Proj and track coords
    proj = compute_projection(griddef)
    ix, iy = get_image_coords(gdf["longitude"].values, gdf["latitude"].values,
                              proj, grid_lons, grid_lats, griddef)

    # Interpolate SIC
    tFB_interp = extract_along_track(tFB_interp, ix, iy, flipud=True)
    sigma_tFB_interp = extract_along_track(sigma_tFB_interp, ix, iy, flipud=True)
    
    # Add to GeoDataFrame
    gdf["tFB_interp"] = tFB_interp
    gdf["sigma_tFB_interp"] = sigma_tFB_interp
    
    return gdf


def compute_apply_flag(sit_product, sic_product, sit_clim_product):
    gdf = sit_product.product
    sic = sic_product.ice_conc
    sit_clim = sit_clim_product.sit_clim
    griddef = {
        "projection": {
            "proj": "laea",
            "ellps": "WGS84",
            "lon_0": 0,
            "lat_0": -90,
            "units": "m"
        },
        "dimension": {
            "n_cols": 432,
            "n_lines": 432,
            "dx": 25000,
            "dy": 25000
        }
    }

    #gdf = gpd.read_file("path_to_your_icesat2_tracks.gpkg")
    merge = add_sic_variables_to_gdf(gdf, sic, griddef)
    merge2 = add_tFB_clim_variables_to_gdf(merge, sit_clim, griddef)
    
    window_size_float = 5000 / 10
    window_size = int(int(window_size_float) // 2 * 2 + 1)
    rolling_kwarg = dict(window=window_size, 
                        min_periods=1,
                        center=True)
    gdf_list = list()
    for beam in merge2['beam'].unique():
        #rolling_merge = merge2[merge2['beam'] == beam].select_dtypes(include='number').rolling(**rolling_kwarg)
        
        # Flag definitions
        sel = merge2[merge2.beam == beam]
        #roll_median = rolling_merge.median()['total_freeboard']
        #roll_std = rolling_merge.std()['total_freeboard']
        #roll_median_layer_flag = rolling_merge.median()['layer_flag']
        #roll_median_rms = rolling_merge.median()['height_segment_rms']
        close_to_ocean = sel['distance_to_ocean_km'] <= 250
        #close_to_ice = sel['distance_to_low_sic_km'] <= 200
        out_of_clim = sel['total_freeboard'] > (sel['tFB_interp'] + 3*sel['sigma_tFB_interp'])
        #flags_bad = roll_median_layer_flag>=1
        #stats_bad = (roll_median > sel['median_15d_track']) & (roll_std > 0.20) & (roll_median_rms <= 0.3)

        exclude_condition = close_to_ocean & out_of_clim #(flags_bad | stats_bad)
        flag = ~exclude_condition | (~close_to_ocean)  
        gdf = sel[(merge2['total_freeboard_quality_flag'] <= 2)][flag]
        gdf_list.append(gdf)
    gdf = pd.concat(gdf_list, ignore_index=True)
    gdf_final = pd.concat(gdf_list).pipe(gpd.GeoDataFrame)
    #Check here the attributes of the gdf_final
    return gdf_final.reset_index(drop=True)
