import numpy as np

from driftaware_sialt.products.sea_ice_concentration import (
    SeaIceConcentrationProducts,
)


def test_interpolation_returns_nan_outside_source_grid():
    ice_conc = {
        'xc': np.array([[0.0, 25_000.0], [0.0, 25_000.0]]),
        'yc': np.array([[25_000.0, 25_000.0], [0.0, 0.0]]),
        'ice_conc': np.array([[100.0, 80.0], [60.0, 40.0]]),
    }

    values = SeaIceConcentrationProducts.interp_ice_concentration(
        ice_conc,
        np.array([12_500.0, 31_250.0]),
        np.array([12_500.0, 12_500.0]),
    )

    assert values[0] == 0.7
    assert np.isnan(values[1])
