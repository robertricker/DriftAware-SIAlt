import datetime

import geopandas as gpd
import netCDF4
import numpy as np
import pytest
from shapely.geometry import Point

from driftaware_sialt.products.sea_ice_thickness import (
    SeaIceThicknessMultiProducts,
)


@pytest.mark.parametrize(
    "target_variable",
    ["sea_ice_thickness", "sea_ice_freeboard", "radar_freeboard"],
)
def test_ers2_uses_cci_l2p_reader(target_variable):
    products = SeaIceThicknessMultiProducts(
        hem="nh",
        sensor=["ers2"],
        target_var=target_variable,
        add_variable=[],
        out_epsg=6931,
    )

    assert ("ers2", target_variable) in products.function_map
    assert products.config["ers2"][target_variable] == {
        "hem_nh": "-NH-",
        "hem_sh": "-SH-",
        "date_str": "{8}",
        "date_pt": "%Y%m%d",
    }


def test_ers2_sample_layout_is_discovered_and_read(tmp_path):
    filename = (
        tmp_path
        / "ESACCI-SEAICE-L2P-SITHICK-RA_ERS2-NH-20021129-fv4p0.nc"
    )
    values = {
        "lat": [75.0, 76.0, 77.0],
        "lon": [10.0, 11.0, 12.0],
        "radar_freeboard": [0.1, 0.2, 0.3],
        "radar_freeboard_uncertainty": [0.01, 0.02, 0.03],
        "sea_ice_freeboard": [0.2, 0.3, 0.4],
        "sea_ice_freeboard_uncertainty": [0.02, 0.03, 0.04],
        "sea_ice_thickness": [1.0, 2.0, np.nan],
        "sea_ice_thickness_uncertainty": [0.1, 0.2, 0.3],
        "snow_depth": [0.2, 0.2, 0.2],
        "time": [1038528000.0, 1038528060.0, 1038528120.0],
    }
    with netCDF4.Dataset(filename, "w") as dataset:
        dataset.createDimension("time", 3)
        flag = dataset.createVariable("flag_miz", "i1", ("time",))
        flag[:] = [0, 2, 0]
        for name, data in values.items():
            variable = dataset.createVariable(name, "f8", ("time",))
            variable[:] = data

    products = SeaIceThicknessMultiProducts(
        hem="nh",
        sensor=["ers2"],
        target_var="sea_ice_thickness",
        add_variable=[],
        out_epsg=6931,
    )
    products.get_file_list({"ers2": str(tmp_path)})
    products.get_file_dates()
    products.get_target_files(
        datetime.datetime(2002, 11, 29),
        datetime.datetime(2002, 11, 30),
    )
    products.get_product(["ers2"])

    assert products.file_list["ers2"] == [str(filename)]
    assert products.file_dates["ers2"] == [datetime.datetime(2002, 11, 29)]
    assert len(products.product) == 1
    assert products.product.iloc[0]["sea_ice_thickness"] == 1.0
    assert products.product.iloc[0]["sea_ice_thickness_l2_unc"] == 0.1
    assert products.product.iloc[0]["sensor"] == "ers2"
    assert products.product.iloc[0]["ers2"] == 1
    assert products.product.crs.to_epsg() == 6931


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
