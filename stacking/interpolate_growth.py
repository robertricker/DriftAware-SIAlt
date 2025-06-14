from gridding import gridding_lib
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy import interpolate
from scipy.interpolate import RBFInterpolator
from scipy.stats import linregress
from shapely.geometry import box
from pyproj import Geod
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from loguru import logger
import sys
#from sklearn.linear_model import RANSACRegressor, LinearRegression


def split_and_compute_area(poly, step=1):
    geod = Geod(ellps="WGS84")
    total_area = 0.0
    for lon_min in range(-180, 180, step):
        lon_max = lon_min + step
        band = box(lon_min, -90, lon_max, 90)
        clipped = poly.intersection(band)
        if not clipped.is_empty:
            # Peut être MultiPolygon si le clipping produit plusieurs morceaux
            if clipped.geom_type == "Polygon":
                area, _ = geod.geometry_area_perimeter(clipped)
                total_area += abs(area)
            elif clipped.geom_type == "MultiPolygon":
                for part in clipped.geoms:
                    area, _ = geod.geometry_area_perimeter(part)
                    total_area += abs(area)
    return total_area / 1e6  # Conversion en km²

def shift_longitude_point(geom):
    if geom.x < 0:
        return type(geom)(geom.x + 360, geom.y)
    return geom

def fix_polygon_wraparound(polygon):
    coords = list(polygon.exterior.coords)
    fixed_coords = []
    for lon, lat in coords:
        # Ramener 360 à 0 pour continuité
        lon_fixed = lon if lon < 180 else lon - 360
        fixed_coords.append((lon_fixed, lat))
    return Polygon(fixed_coords)

def interpolate_growth_gridd(data, interp_var, growth_range, grid, cell_width, min_n_tiepoints, nbs, hem):
    merged = gpd.sjoin(data, grid, how='left', predicate='within')
    tmp = (merged.groupby(['index_right', 'dt_days'], as_index=False)
           .agg({'geometry': 'first', interp_var: 'mean'})
           .pipe(gpd.GeoDataFrame, geometry='geometry', crs=merged.crs))
    n_tiepoints = tmp.groupby('index_right')['dt_days'].count()
    valid_indices = n_tiepoints[n_tiepoints >= min_n_tiepoints].index
    tmp = tmp[tmp['index_right'].isin(valid_indices)]
    tmp.set_index('index_right', inplace=True)
    eps = 1.8
    lat_range = [40.0, 90.0] if hem == 'nh' else [-40.0, -90.0]
    fsm = interpolate.interp1d(np.array(lat_range), np.array([80, 10]))
    #density = df.groupby('lat')['nombre de points'].sum().reset_index()
    # perform linear fit
    tmp['coeff'] = tmp.groupby('index_right').apply(
        lambda x: np.polyfit(x['dt_days'], x[interp_var], deg=1, cov=True))
    tmp['growth'] = [x[0][0] for x in tmp['coeff'].values]
    tmp['growth_unc'] = [np.sqrt(np.diag(x[1])[0]) for x in tmp['coeff'].values]
    tmp[(tmp['growth'] > growth_range[1]) | (tmp['growth'] < growth_range[0])] = np.nan
    growth_raw = pd.merge(
        merged, tmp[~tmp.index.duplicated(keep='first')].reset_index(),
        left_on='index_right', right_on='index_right', how='left')['growth']
    tmp = tmp.drop(columns=['coeff'])
    tmp.index.names = ['index']

    # gridding of growth coefficients
    growth_grid = gridding_lib.grid_data(tmp, grid, ['growth', 'growth_unc'], ['growth', 'growth_unc'], fill_nan=True)
    centroidseries = growth_grid['geometry'].centroid
    growth_grid['yc'], growth_grid['xc'] = centroidseries.x, centroidseries.y

    # interpolation of growth for all valid target variable data points
    fg = RBFInterpolator(np.vstack((np.array(growth_grid.dropna()['yc']),
                                    np.array(growth_grid.dropna()['xc']))).transpose(),
                         np.array(growth_grid.dropna()['growth']),
                         neighbors=nbs,
                         smoothing=fsm(
                             np.array(growth_grid.dropna().geometry.centroid.to_crs(4326).geometry.y)),
                         kernel='gaussian', epsilon=eps/cell_width)
    
    fg_s10 = RBFInterpolator(np.vstack((np.array(growth_grid.dropna()['yc']),
                                    np.array(growth_grid.dropna()['xc']))).transpose(),
                         np.array(growth_grid.dropna()['growth']),
                         neighbors=nbs,
                         smoothing=10,
                         kernel='gaussian', epsilon=eps/cell_width)
    fg_nb = RBFInterpolator(np.vstack((np.array(growth_grid.dropna()['yc']),
                                       np.array(growth_grid.dropna()['xc']))).transpose(),
                         np.array(growth_grid.dropna()['growth']),
                         neighbors=nbs/10,
                         smoothing=fsm(
                             np.array(growth_grid.dropna().geometry.centroid.to_crs(4326).geometry.y)),
                         kernel='gaussian', epsilon=eps/cell_width)
    fg_unc = RBFInterpolator(np.vstack((np.array(growth_grid.dropna()['yc']),
                                        np.array(growth_grid.dropna()['xc']))).transpose(),
                             np.array(growth_grid.dropna()['growth_unc']),
                             neighbors=nbs,
                             smoothing=fsm(
                                 np.array(growth_grid.dropna().geometry.centroid.to_crs(4326).geometry.y)),
                             kernel='gaussian', epsilon=eps/cell_width)
    return fg, fg_nb, fg_s10, fg_unc, growth_raw.values, n_tiepoints

def compute_area_km2(poly):
    geod = Geod(ellps="WGS84")
    if poly.is_empty or poly.geom_type != "Polygon":
        return 0.0
    area, _ = geod.geometry_area_perimeter(poly)
    return abs(area) / 1e6

def interpolate_growth(data, interp_var, growth_range, grid, cell_width, min_n_tiepoints, nbs, hem):
    merged = gpd.sjoin(data, grid, how='left', predicate='within')
    tmp = (merged.groupby(['index_right', 'dt_days'], as_index=False)
           .agg({'geometry': 'first', interp_var: 'mean'})
           .pipe(gpd.GeoDataFrame, geometry='geometry', crs=merged.crs))
    n_tiepoints = tmp.groupby('index_right')['dt_days'].count()
    valid_indices = n_tiepoints[n_tiepoints >= min_n_tiepoints].index
    if len(valid_indices) == []:
        logger.error('Number of tie points insufficiant for all the grid cell')
        sys.exit()
    tmp = tmp[tmp['index_right'].isin(valid_indices)]
    tmp.set_index('index_right', inplace=True)
    eps = 1.8
    lat_range = [40.0, 90.0] if hem == 'nh' else [-40.0, -90.0]
    gdf = data.to_crs("EPSG:4326")
    #gdf["geometry"] = gdf["geometry"].apply(shift_longitude_point)
    gdf["lat_band"] = gdf.geometry.y.round()
    counts = gdf.groupby("lat_band").size().reset_index(name="count")
    counts["geometry"] = counts["lat_band"].apply(lambda lat: box(-180, lat, 180, lat + 1))
    
    bands_gdf = gpd.GeoDataFrame(counts, geometry="geometry", crs="EPSG:4326")
    geod = Geod(ellps="WGS84")
    world = gpd.read_file("/cluster/projects/108541-SO-SIMBA/data_tmp/input_data/auxdata/land_mask_cartopy/ne_110m_land.shp")
    land_gdf = world[world['geometry'].intersects(counts.geometry[0])]

    #ocean_polygon_without_land = counts['geometry'].difference(land_gdf.unary_union)
    diff_geoms = counts['geometry'].apply(lambda geom: geom.difference(land_gdf.unary_union))

    # Clean and get valid geometries
    clean_geoms = []
    for geom in diff_geoms:
        if isinstance(geom, (Polygon, MultiPolygon)):
            clean_geoms.append(geom)
        elif isinstance(geom, GeometryCollection):
            clean_geoms.extend([g for g in geom if isinstance(g, (Polygon, MultiPolygon))])

    ocean_without_land_gdf = gpd.GeoDataFrame(geometry=clean_geoms, crs="EPSG:4326")

    
    counts["area_km2"] = counts.geometry.apply(split_and_compute_area)
    counts["area_km2_no_land"] = ocean_without_land_gdf.geometry.apply(split_and_compute_area)

    counts["density_km2"] = counts["count"] / counts["area_km2"]
    counts["density_km2_no_land"] = counts["count"] / counts["area_km2_no_land"]

    slope, intercept, *_ = linregress(counts['lat_band'].values, counts['density_km2_no_land'].values)
    fsm = interpolate.interp1d([0, 0.035], np.array([80, 10])) # Notebook and SOSIMBA ATBD
    
    # perform linear fit
    tmp['coeff'] = tmp.groupby('index_right').apply(
        lambda x: np.polyfit(x['dt_days'], x[interp_var], deg=1, cov=True))
    tmp['growth'] = [x[0][0] for x in tmp['coeff'].values]
    tmp['growth_unc'] = [np.sqrt(np.diag(x[1])[0]) for x in tmp['coeff'].values]
    tmp[(tmp['growth'] > growth_range[1]) | (tmp['growth'] < growth_range[0])] = np.nan
    growth_raw = pd.merge(
        merged, tmp[~tmp.index.duplicated(keep='first')].reset_index(),
        left_on='index_right', right_on='index_right', how='left')['growth']
    tmp = tmp.drop(columns=['coeff'])
    tmp.index.names = ['index']

    # gridding of growth coefficients
    growth_grid = gridding_lib.grid_data(tmp, grid, ['growth', 'growth_unc'], ['growth', 'growth_unc'], fill_nan=True)
    centroidseries = growth_grid['geometry'].centroid
    growth_grid['yc'], growth_grid['xc'] = centroidseries.x, centroidseries.y
    arr_density = slope*(np.array(growth_grid.dropna().geometry.centroid.to_crs(4326).geometry.y))+intercept
    arr_positif = np.where(arr_density < 0, 0, arr_density)
    # interpolation of growth for all valid target variable data points
    fg = RBFInterpolator(np.vstack((np.array(growth_grid.dropna()['yc']),
                                    np.array(growth_grid.dropna()['xc']))).transpose(),
                         np.array(growth_grid.dropna()['growth']),
                         neighbors=nbs,
                         smoothing=fsm(arr_positif),
                         kernel='gaussian', epsilon=eps/cell_width)
    
    fg_unc = RBFInterpolator(np.vstack((np.array(growth_grid.dropna()['yc']),
                                        np.array(growth_grid.dropna()['xc']))).transpose(),
                             np.array(growth_grid.dropna()['growth_unc']),
                             neighbors=nbs,
                             smoothing=fsm(arr_positif),
                             kernel='gaussian', epsilon=eps/cell_width)
    return fg, fg_unc, growth_raw.values, n_tiepoints, counts
