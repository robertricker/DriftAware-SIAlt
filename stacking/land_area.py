"""Latitude-band ocean areas for density-dependent growth interpolation."""

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely import make_valid
from shapely.geometry import Polygon, box


POLAR_CRS = {"nh": "EPSG:6931", "sh": "EPSG:6932"}


def latitude_band_polygon(lat_band, hemisphere, samples=720):
    """Build a one-degree latitude band in a polar equal-area projection."""
    if hemisphere not in POLAR_CRS:
        raise ValueError("hemisphere must be 'nh' or 'sh'")
    if not -90 <= lat_band < 90:
        raise ValueError(f"latitude band must be in [-90, 90): {lat_band}")

    lat_min = float(lat_band)
    lat_max = lat_min + 1.0
    if hemisphere == "nh":
        equatorward_lat, poleward_lat = lat_min, lat_max
    else:
        equatorward_lat, poleward_lat = lat_max, lat_min

    longitudes = np.linspace(-180.0, 180.0, samples + 1)
    transformer = Transformer.from_crs("EPSG:4326", POLAR_CRS[hemisphere], always_xy=True)
    outer_x, outer_y = transformer.transform(
        longitudes, np.full_like(longitudes, equatorward_lat))
    outer = list(zip(outer_x, outer_y))

    if abs(poleward_lat) == 90.0:
        polygon = Polygon(outer)
    else:
        inner_x, inner_y = transformer.transform(
            longitudes, np.full_like(longitudes, poleward_lat))
        inner = list(reversed(list(zip(inner_x, inner_y))))
        polygon = Polygon(outer, holes=[inner])

    return polygon if polygon.is_valid else polygon.buffer(0)


@lru_cache(maxsize=4)
def load_land_union(path, hemisphere):
    """Load and cache a global land mask in the requested polar CRS."""
    mask_path = Path(path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"land mask does not exist: {mask_path}")
    if hemisphere not in POLAR_CRS:
        raise ValueError("hemisphere must be 'nh' or 'sh'")

    land = gpd.read_file(mask_path)
    if land.crs is None:
        raise ValueError(f"land mask has no coordinate reference system: {mask_path}")

    land = land.to_crs("EPSG:4326")
    hemisphere_extent = box(-180, 0, 180, 90) if hemisphere == "nh" else box(-180, -90, 180, 0)
    clipped = land.geometry.apply(lambda geometry: geometry.intersection(hemisphere_extent))
    clipped = clipped[~clipped.is_empty]
    if clipped.empty:
        raise ValueError(f"land mask contains no geometry for hemisphere {hemisphere}: {mask_path}")

    clipped = clipped.apply(make_valid)
    polar_land = gpd.GeoSeries(clipped, crs="EPSG:4326").to_crs(POLAR_CRS[hemisphere])
    polar_land = polar_land.apply(make_valid)
    land_union = polar_land.unary_union
    return land_union if land_union.is_valid else land_union.buffer(0)


def calculate_band_areas(band, land_union=None):
    """Return total, land, and ocean areas in square kilometres."""
    total_area = band.area / 1e6
    land_area = 0.0
    if land_union is not None and band.intersects(land_union):
        land_area = min(band.intersection(land_union).area / 1e6, total_area)
    ocean_area = max(total_area - land_area, 0.0)
    # Suppress tiny overlay slivers at the polar point and coastline boundaries.
    if total_area > 0 and ocean_area / total_area < 1e-5:
        ocean_area = 0.0
        land_area = total_area
    return total_area, land_area, ocean_area


def latitude_band_density(data, hemisphere, land_mask_path=None, exclude_land=False):
    """Count points and calculate total/ocean area for occupied latitude bands."""
    if data.crs is None:
        raise ValueError("growth input data must have a coordinate reference system")

    points = data[["geometry"]].copy().to_crs("EPSG:4326")
    latitudes = points.geometry.y.to_numpy()
    points["lat_band"] = np.floor(latitudes).clip(-90, 89).astype(int)
    counts = points.groupby("lat_band").size().reset_index(name="count")

    land_union = None
    if exclude_land:
        if not land_mask_path:
            raise ValueError("a land-mask path is required when land-area correction is enabled")
        land_union = load_land_union(str(land_mask_path), hemisphere)

    records = []
    for row in counts.itertuples(index=False):
        band = latitude_band_polygon(row.lat_band, hemisphere)
        total_area, land_area, ocean_area = calculate_band_areas(band, land_union)
        records.append({
            "lat_band": row.lat_band,
            "count": row.count,
            "area_km2": total_area,
            "land_area_km2": land_area,
            "area_km2_no_land": ocean_area,
            "density_km2": row.count / total_area if total_area > 0 else np.nan,
            "density_km2_no_land": row.count / ocean_area if ocean_area > 0 else np.nan,
            "geometry": band,
        })

    return gpd.GeoDataFrame(records, geometry="geometry", crs=POLAR_CRS[hemisphere])
