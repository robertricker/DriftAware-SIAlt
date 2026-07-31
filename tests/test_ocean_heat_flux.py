import numpy as np
import pytest

from driftaware_sialt.products.ocean_heat_flux import OceanHeatFluxProducts


def test_interpolate_ocean_heat_flux():
    product = OceanHeatFluxProducts(
        hem="nh",
        product_id="sose",
        out_epsg="EPSG:6931")
    xc, yc = np.meshgrid([0.0, 2.0], [0.0, 2.0])
    product.ohf = {
        "xc": xc,
        "yc": yc,
        "ohf": np.array([[[0.0, 2.0], [2.0, 4.0]]]),
    }

    result = product.interp_ocean_heat_flux(
        np.array([1.0]),
        np.array([1.0]))

    assert result[0] == pytest.approx(2.0)
