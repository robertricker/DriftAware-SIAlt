import geopandas as gpd
from shapely.geometry import Point

from driftaware_sialt.products.sea_ice_thickness import (
    SeaIceThicknessMultiProducts,
)


def test_get_product_does_not_reuse_unavailable_sensor_from_previous_date():
    products = SeaIceThicknessMultiProducts(
        hem="nh",
        sensor=["cryosat2", "sentinel3a"],
        target_var="sea_ice_thickness",
        add_variable=[],
        out_epsg=6931)

    def make_product(sensor):
        return gpd.GeoDataFrame(
            {"sea_ice_thickness": [1.0]},
            geometry=[Point(0, 0)],
            crs="EPSG:6931")

    products.function_map[
        ("cryosat2", "sea_ice_thickness")] = make_product
    products.function_map[
        ("sentinel3a", "sea_ice_thickness")] = make_product

    products.get_product(["cryosat2", "sentinel3a"])
    products.get_product(["cryosat2"])

    assert products.product["sensor"].tolist() == ["cryosat2"]
    assert "sentinel3a" not in products.product.columns
