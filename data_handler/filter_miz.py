import numpy as np
import netCDF4
import glob
import re
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

def interp_var(gridvar, gridvar_x, gridvar_y, x, y):
    from scipy.interpolate import RegularGridInterpolator
    
    xc, yc = gridvar_x[0, :], gridvar_y[:, 0]
    # Check if xc and yc are in descending order
    if xc[0] > xc[-1]:
        xc = xc[::-1]
        gridvar = gridvar[:, ::-1]
    if yc[0] > yc[-1]:
        yc = yc[::-1]
        gridvar = gridvar[::-1, :]

    interp_func = RegularGridInterpolator((xc, yc), gridvar.T, method='linear')
    interp = interp_func((x, y))
    return interp

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

    # Mask invalid
    ice_conc = np.where(ice_conc < 0, np.nan, ice_conc)

    # Proj and track coords
    proj = compute_projection(griddef)
    gdf_lonlat = gdf.to_crs("EPSG:4326")
    
    gdf["longitude"] = gdf_lonlat.geometry.x
    gdf["latitude"] = gdf_lonlat.geometry.y

    sic_track = interp_var(ice_conc, ds_ice_conc["xc"], ds_ice_conc["yc"], gdf.geometry.x, gdf.geometry.y)

    # Compute distances
    dx = float(griddef["dimension"]["dx"]) # can be determined without the griddef
    dist_to_ocean = compute_proximity(ice_conc, 15., dx)
    dist_to_low_sic = compute_proximity(ice_conc, 70., dx)

    # Interpolate distances
    dist_ocean_track = interp_var(dist_to_ocean, ds_ice_conc["xc"], ds_ice_conc["yc"], gdf.geometry.x, gdf.geometry.y)
    dist_low_sic_track = interp_var(dist_to_low_sic, ds_ice_conc["xc"], ds_ice_conc["yc"], gdf.geometry.x, gdf.geometry.y)

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

    # Proj and track coords
    gdf_lonlat = gdf.to_crs("EPSG:4326")

    gdf["longitude"] = gdf_lonlat.geometry.x
    gdf["latitude"] = gdf_lonlat.geometry.y

    # Interpolate SIT
    clim_interp = interp_var(clim_interp, ds_clim_product["xc"], ds_clim_product["yc"], gdf.geometry.x, gdf.geometry.y)
    sigma_clim_interp = interp_var(sigma_clim_interp, ds_clim_product["xc"], ds_clim_product["yc"], gdf.geometry.x, gdf.geometry.y)
    
    # Add to GeoDataFrame
    gdf["clim_interp"] = clim_interp
    gdf["sigma_clim_interp"] = sigma_clim_interp
    
    return gdf


def add_rfb_from_dir_to_gdf(gdf, rfb_dir, rfb_var='radar_freeboard', crs='epsg:6932'):
    """
    Load an rFB netCDF from a directory (first match) and interpolate the radar freeboard
    onto the GeoDataFrame points. Adds column `radar_freeboard` to `gdf`.

    Parameters
    - gdf: GeoDataFrame with geometry in projection `crs` or EPSG:4326 coordinates
    - rfb_dir: directory containing rFB netCDF files
    - rfb_var: variable name in netCDF to read (default 'radar_freeboard')
    - crs: target CRS string for interpolation (default 'epsg:6932')

    Returns updated GeoDataFrame.
    """
    files = sorted(glob.glob(rfb_dir + '/*.nc')) if rfb_dir else []
    if not files:
        return gdf

    try:
        ds = netCDF4.Dataset(files[0])
    except Exception:
        return gdf

    # try to find variable name
    varname = rfb_var if rfb_var in ds.variables else None
    if varname is None:
        for candidate in ['radar_freeboard', 'rfb', 'RFB']:
            if candidate in ds.variables:
                varname = candidate
                break
    if varname is None:
        ds.close()
        return gdf

    lon = np.ma.getdata(ds.variables['lon'][:, :]).flatten()
    lat = np.ma.getdata(ds.variables['lat'][:, :]).flatten()
    values = np.ma.getdata(ds.variables[varname][:, :]).flatten()

    # build grid coords
    xc = np.ma.getdata(ds.variables['xc'][:])
    yc = np.ma.getdata(ds.variables['yc'][:])
    xc_m, yc_m = np.meshgrid(xc, yc)
    coords = np.transpose(np.vstack((lon, lat)))

    # Transform lon/lat to projection of the grid used by gdf
    from io_tools import transform_coords
    xg, yg = transform_coords(lon, lat, 'epsg:4326', crs)

    # Interpolate onto the xc,yc grid
    from scipy.interpolate import griddata
    try:
        rfb_grid = griddata(np.transpose(np.vstack((xg, yg))), values, (xc_m, yc_m), method='nearest')
    except Exception:
        ds.close()
        return gdf

    # Interpolate grid values to track points
    try:
        rfb_track = interp_var(rfb_grid, xc_m, yc_m, gdf.geometry.x, gdf.geometry.y)
    except Exception:
        ds.close()
        return gdf

    ds.close()

    gdf['radar_freeboard'] = rfb_track
    return gdf

def get_coord(ds, short_name, standard_name):
    candidates = [short_name, standard_name, short_name.upper(), standard_name.capitalize()]
    for name in candidates:
        if name in ds.variables:
            return ds.variables[name]
    # Fallback sur standard_name (convention CF)
    for name, var in ds.variables.items():
        if getattr(var, 'standard_name', None) == standard_name:
            return var
    raise KeyError(f"Coordonnée '{standard_name}' introuvable. "
                    f"Variables disponibles : {list(ds.variables.keys())}")


def _climatology_files_by_day(files):
    """Return the climatology files indexed by their ``MMDD`` suffix."""
    climatology_files = {}
    for file in files:
        match = re.search(r'_(\d{2})(\d{2})\.nc$', file)
        if match:
            climatology_files[match.group(1) + match.group(2)] = file
    return climatology_files


def _reference_days(gdf, reference_date=None):
    """Get an ``MMDD`` climatology key for every row of a GeoDataFrame."""
    if reference_date is not None:
        return np.full(len(gdf), pd.Timestamp(reference_date).strftime('%m%d'), dtype=object)
    if 'time' not in gdf.columns:
        return None

    time = gdf['time']
    if pd.api.types.is_numeric_dtype(time):
        dates = pd.to_datetime(time, unit='s', errors='coerce', utc=True)
    else:
        dates = pd.to_datetime(time, errors='coerce', utc=True)
    return dates.dt.strftime('%m%d').to_numpy()


def _interpolate_reference_file(gdf, file, reference_var, crs, reference_sigma_var):
    """Interpolate one reference NetCDF onto the rows in ``gdf``."""
    try:
        ds = netCDF4.Dataset(file)
    except Exception as error:
        print(f'ERROR: cannot open reference file {file}: {error}')
        return None, None

    try:
        varname = reference_var if reference_var in ds.variables else None
        if varname is None:
            for candidate in [reference_var, 'radar_freeboard', 'rfb', 'RFB']:
                if candidate in ds.variables:
                    varname = candidate
                    break
        if varname is None:
            print(f'WARNING: {reference_var} is absent from {file}')
            return None, None

        lon_var = get_coord(ds, 'lon', 'longitude')
        lat_var = get_coord(ds, 'lat', 'latitude')
        lon = np.ma.getdata(lon_var[:, :]).flatten()
        lat = np.ma.getdata(lat_var[:, :]).flatten()
        values = np.ma.getdata(ds.variables[varname][:, :]).flatten()
        xc = np.ma.getdata(ds.variables['xc'][:])
        yc = np.ma.getdata(ds.variables['yc'][:])
        xc_m, yc_m = np.meshgrid(xc, yc)

        from io_tools import transform_coords
        from scipy.interpolate import griddata
        xg, yg = transform_coords(lon, lat, 'epsg:4326', crs)
        ref_grid = griddata(np.transpose(np.vstack((xg, yg))), values,
                           (xc_m, yc_m), method='nearest')
        ref_track = interp_var(ref_grid, xc_m, yc_m,
                               gdf.geometry.x, gdf.geometry.y)

        sigma_track = None
        if reference_sigma_var and reference_sigma_var in ds.variables:
            sig_values = np.ma.getdata(ds.variables[reference_sigma_var][:, :]).flatten()
            sig_grid = griddata(np.transpose(np.vstack((xg, yg))), sig_values,
                               (xc_m, yc_m), method='nearest')
            sigma_track = interp_var(sig_grid, xc_m, yc_m,
                                     gdf.geometry.x, gdf.geometry.y)
        return ref_track, sigma_track
    finally:
        ds.close()


def add_reference_var_to_gdf(gdf, reference_dir=None, reference_var='radar_freeboard', crs='epsg:6932',
                             reference_sigma_var=None, reference_scale=1/0.42, reference_date=None):
    """
    Load an auxiliary reference field and interpolate it onto trajectory points.

    A directory containing files ending in ``_MMDD.nc`` is treated as a daily
    climatology.  In that case, each row receives the field from the file for
    its calendar day (from ``reference_date``, or from the ``time`` column).
    Other reference directories retain the historical behaviour: first file.
    """
    files = sorted(glob.glob(reference_dir + '/**/*.nc', recursive=True)) if reference_dir else []
    if not files:
        return gdf

    climatology_files = _climatology_files_by_day(files)
    if not climatology_files:
        print(f'INFO: Using reference file {files[0]}')
        ref_track, sigma_track = _interpolate_reference_file(
            gdf, files[0], reference_var, crs, reference_sigma_var)
        if ref_track is not None:
            gdf[reference_var] = ref_track
            if sigma_track is not None:
                gdf[reference_sigma_var] = sigma_track
        return gdf

    days = _reference_days(gdf, reference_date)
    if days is None:
        print('WARNING: climatology detected but no reference_date or time column was provided')
        return gdf

    reference_values = np.full(len(gdf), np.nan)
    sigma_values = np.full(len(gdf), np.nan)
    for day in np.unique(days[pd.notna(days)]):
        positions = np.flatnonzero(days == day)
        file = climatology_files.get(day)
        if file is None:
            print(f'WARNING: no climatology file found for {day}')
            continue
        print(f'INFO: Using climatology file {file}')
        ref_track, sigma_track = _interpolate_reference_file(
            gdf.iloc[positions], file, reference_var, crs, reference_sigma_var)
        if ref_track is not None:
            reference_values[positions] = ref_track
        if sigma_track is not None:
            sigma_values[positions] = sigma_track

    gdf[reference_var] = reference_values
    if reference_sigma_var and np.any(~np.isnan(sigma_values)):
        gdf[reference_sigma_var] = sigma_values
    return gdf


def compute_apply_flag_reference(gdf, sic_product, target_var, sensor, crs='epsg:6932',
                                reference_var='radar_freeboard', reference_dir=None,
                                reference_sigma_var=None, reference_scale=1/0.42,
                                reference_date=None):
    """Alternative MIZ filter using a generic reference field instead of climatology."""
    sic = sic_product.ice_conc
    griddef = {
        "projection": {"proj": "laea", "ellps": "WGS84", "lon_0": 0, "lat_0": -90, "units": "m"},
        "dimension": {"n_cols": 432, "n_lines": 432, "dx": 25000, "dy": 25000}
    }

    merge = add_sic_variables_to_gdf(gdf, sic, griddef, sensor, crs)
    merge = add_reference_var_to_gdf(merge, reference_dir=reference_dir, reference_var=reference_var,
                                    crs=crs, reference_sigma_var=reference_sigma_var,
                                    reference_scale=reference_scale,
                                    reference_date=reference_date)

    if reference_var not in merge.columns:
        return gdf

    # convert the reference field to the same quantity used by the classical MIZ check
    # e.g. radar freeboard -> total freeboard if needed
    merge['clim_interp'] = merge[reference_var] * float(reference_scale)
    if reference_sigma_var and reference_sigma_var in merge.columns:
        merge['sigma_clim_interp'] = merge[reference_sigma_var] * float(reference_scale)
    else:
        merge['sigma_clim_interp'] = 0.0

    window_size_float = 5000 / 10
    window_size = int(int(window_size_float) // 2 * 2 + 1)
    rolling_kwarg = dict(window=window_size, min_periods=1, center=True)

    if sensor == 'icesat2':
        gdf_list = []
        for beam in merge['beam'].unique():
            sel = merge[merge.beam == beam]
            not_coastal_weddell = ~((sel['longitude'] > -60) & (sel['longitude'] < -45))
            close_to_ocean = sel['distance_to_ocean_km'] <= 300
            out_of_ref = sel[f'{target_var}'] > (sel['clim_interp'] + sel['sigma_clim_interp'])
            exclude_condition = close_to_ocean & out_of_ref & not_coastal_weddell
            flag = ~exclude_condition | (~close_to_ocean) 
            gdf_list.append(sel[flag])
        gdf_final = pd.concat(gdf_list, ignore_index=True).pipe(gpd.GeoDataFrame).reset_index(drop=True)
    else:
        not_coastal_weddell = ~((merge['longitude'] > -60) & (merge['longitude'] < -45))
        out_of_ref = merge[f'{target_var}'] > (merge['clim_interp'] + merge['sigma_clim_interp'])
        close_to_ocean = merge['distance_to_ocean_km'] <= 300
        exclude_condition = close_to_ocean & out_of_ref & not_coastal_weddell
        flag = ~exclude_condition | (~close_to_ocean)
        gdf_final = merge[flag]
    return gdf_final


# Keep the historical climatology-based function unchanged.
def compute_apply_flag(gdf, sic_product, sit_clim_product, target_var, sensor, crs='epsg:6932',
                       use_rfb=False, rfb_col='radar_freeboard', rfb_sigma_col=None, rfb_scale=1/0.42):
    
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

    # Optional alternative path retained for compatibility.
    if use_rfb:
        if rfb_col in merge2.columns:
            try:
                merge2['clim_interp'] = merge2[rfb_col] * float(rfb_scale)
            except Exception:
                merge2['clim_interp'] = merge2[rfb_col] / 0.42
            if rfb_sigma_col and (rfb_sigma_col in merge2.columns):
                merge2['sigma_clim_interp'] = merge2[rfb_sigma_col] * float(rfb_scale)
            elif 'sigma_clim_interp' in merge2.columns:
                merge2['sigma_clim_interp'] = merge2['sigma_clim_interp'] * float(rfb_scale)
            else:
                merge2['sigma_clim_interp'] = np.nan

    window_size_float = 5000 / 10
    window_size = int(int(window_size_float) // 2 * 2 + 1)
    rolling_kwarg = dict(window=window_size, 
                        min_periods=1,
                        center=True)
    if sensor == 'icesat2':  
        gdf_list = list()
        for beam in merge2['beam'].unique():
            sel = merge2[merge2.beam == beam]
            not_coastal_weddell = ~((sel['longitude'] > -60) & (sel['longitude'] < -45)) 
            close_to_ocean = sel['distance_to_ocean_km'] <= 300
            out_of_clim = sel[f'{target_var}'] > (sel['clim_interp'] + sel['sigma_clim_interp'])
            exclude_condition = close_to_ocean & out_of_clim & not_coastal_weddell
            flag = ~exclude_condition | (~close_to_ocean)  
            gdf = sel[flag]
            gdf_list.append(gdf)
            gdf = pd.concat(gdf_list, ignore_index=True)
            gdf_final = pd.concat(gdf_list).pipe(gpd.GeoDataFrame).reset_index(drop=True)
    else:
        not_coastal_weddell = ~((merge2['longitude'] > -60) & (merge2['longitude'] < -45))
        out_of_clim = merge2[f'{target_var}'] > (merge2['clim_interp'] + merge2['sigma_clim_interp'])
        close_to_ocean = merge2['distance_to_ocean_km'] <= 300
        exclude_condition = close_to_ocean & out_of_clim & not_coastal_weddell
        flag = ~exclude_condition | (~close_to_ocean)  
        gdf_final = merge2[flag]
    return gdf_final
