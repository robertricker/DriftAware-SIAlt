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

def add_sic_variables_to_gdf(gdf, ds_ice_conc, griddef, sensor, crs):
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
    ice_conc = ds_ice_conc["ice_conc_no_0"][:, :]
    grid_lons = ds_ice_conc["lon"]
    grid_lats = ds_ice_conc["lat"]

    # Mask invalid
    ice_conc = np.where(ice_conc < 0, np.nan, ice_conc)

    # Proj and track coords
    proj = compute_projection(griddef)
    gdf_lonlat = gdf.to_crs("EPSG:4326")
    if "icesat2" in sensor:
        gdf["longitude"] = gdf_lonlat.geometry.x
        gdf["latitude"] = gdf_lonlat.geometry.y
    else:
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(gdf.reset_index().x, 
                                                              gdf.reset_index().y),crs="EPSG:6932" )
        
        gdf["longitude"][:] = points.to_crs('EPSG:4326').geometry.x
        gdf["latitude"][:] = points.to_crs('EPSG:4326').geometry.y
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

def add_tFB_clim_variables_to_gdf(gdf, ds_clim_product, griddef, sensor, crs, target_var):
    # Load clim tFB grid
    if target_var == "total_freeboard":
        clim_interp = ds_clim_product["tFB_interp"]
        sigma_clim_interp = ds_clim_product["sigma_tFB_interp"]
    elif target_var == "sea_ice_thickness":
        clim_interp = ds_clim_product["SIT_interp"]
        sigma_clim_interp = ds_clim_product["sigma_SIT_interp"]
    grid_lons = ds_clim_product["longitude"]
    grid_lats = ds_clim_product["latitude"]

    # Proj and track coords
    proj = compute_projection(griddef)
    gdf_lonlat = gdf.to_crs("EPSG:4326")
    if sensor == "icesat2":
        gdf["longitude"] = gdf_lonlat.geometry.x
        gdf["latitude"] = gdf_lonlat.geometry.y
    else:
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(gdf.reset_index().x, 
                                                              gdf.reset_index().y),crs="EPSG:6932" )
        
        gdf["longitude"] = points.to_crs('EPSG:4326').geometry.x
        gdf["latitude"] = points.to_crs('EPSG:4326').geometry.y
    ix, iy = get_image_coords(gdf["longitude"].values, gdf["latitude"].values,
                              proj, grid_lons, grid_lats, griddef)

    # Interpolate SIC
    clim_interp = extract_along_track(clim_interp, ix, iy, flipud=True)
    sigma_clim_interp = extract_along_track(sigma_clim_interp, ix, iy, flipud=True)
    
    # Add to GeoDataFrame
    gdf["clim_interp"] = clim_interp
    gdf["sigma_clim_interp"] = sigma_clim_interp
    
    return gdf


def compute_apply_flag(gdf, sic_product, sit_clim_product, target_var, sensor, crs='epsg:6932'):
    
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

    merge = add_sic_variables_to_gdf(gdf, sic, griddef, sensor, crs)
    merge2 = add_tFB_clim_variables_to_gdf(merge, sit_clim, griddef, sensor, crs, target_var)
    
    window_size_float = 5000 / 10
    window_size = int(int(window_size_float) // 2 * 2 + 1)
    rolling_kwarg = dict(window=window_size, 
                        min_periods=1,
                        center=True)
    if sensor == 'icesat2':  
        gdf_list = list()
        for beam in merge2['beam'].unique():
            #rolling_merge = merge2[merge2['beam'] == beam].select_dtypes(include='number').rolling(**rolling_kwarg)
            
            # Flag definitions
            sel = merge2[merge2.beam == beam]
            #roll_median = rolling_merge.median()['total_freeboard']
            #roll_std = rolling_merge.std()['total_freeboard']
            #roll_median_layer_flag = rolling_merge.median()['layer_flag']
            #roll_median_rms = rolling_merge.median()['height_segment_rms']
            not_coastal_weddell = ~((sel['longitude'] > -60) & (sel['longitude'] < -45)) 
            close_to_ocean = sel['distance_to_ocean_km'] <= 300
            #close_to_ice = sel['distance_to_low_sic_km'] <= 200
            out_of_clim = sel[f'{target_var}'] > (sel['clim_interp'] + sel['sigma_clim_interp'])
            #flags_bad = roll_median_layer_flag>=1
            #stats_bad = (roll_median > sel['median_15d_track']) & (roll_std > 0.20) & (roll_median_rms <= 0.3)

            exclude_condition = close_to_ocean & out_of_clim & not_coastal_weddell #(flags_bad | stats_bad)
            flag = ~exclude_condition | (~close_to_ocean)  
            #gdf = sel[(merge2['total_freeboard_quality_flag'] <= 2)][flag]
            gdf = sel[flag]
            gdf_list.append(gdf)
            gdf = pd.concat(gdf_list, ignore_index=True)
            gdf_final = pd.concat(gdf_list).pipe(gpd.GeoDataFrame).reset_index(drop=True)
    else:
        not_coastal_weddell = ~((merge2['longitude'] > -60) & (merge2['longitude'] < -45))
        out_of_clim = merge2[f'{target_var}'] > (merge2['clim_interp'] + merge2['sigma_clim_interp'])
        close_to_ocean = merge2['distance_to_ocean_km'] <= 300
        exclude_condition = close_to_ocean & out_of_clim & not_coastal_weddell #(flags_bad | stats_bad)
        flag = ~exclude_condition | (~close_to_ocean)  
        gdf_final = merge2[flag]
    #Check here the attributes of the gdf_final
    return gdf_final
