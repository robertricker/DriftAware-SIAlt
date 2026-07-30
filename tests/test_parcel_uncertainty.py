import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from driftaware_sialt.stacking.drift_aware_processor import (
    grid_parcel_uncertainty,
)


@pytest.mark.parametrize(
    "target_var",
    [
        "sea_ice_thickness",
        "sea_ice_freeboard",
        "radar_freeboard",
        "total_freeboard",
    ])
def test_grid_parcel_uncertainty_from_l2_uncertainty(target_var):
    source_unc_var = target_var + "_l2_unc"
    parcel_unc_var = target_var + "_parcel_unc"
    data = gpd.GeoDataFrame(
        {source_unc_var: [0.3, 0.4]},
        geometry=[Point(-0.5, 0), Point(0.5, 0)],
        crs="EPSG:6931")
    grid = gpd.GeoDataFrame(
        geometry=[box(-1, -1, 1, 1)],
        crs=data.crs)

    parcel_uncertainty = grid_parcel_uncertainty(
        data, grid, source_unc_var, parcel_unc_var)

    assert parcel_uncertainty.iloc[0] == pytest.approx(0.25)
    assert data[source_unc_var].tolist() == [0.3, 0.4]
