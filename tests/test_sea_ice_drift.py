import numpy as np

from driftaware_sialt.products.sea_ice_drift import _coastal_drift_taper


def test_coastal_drift_taper_is_linear_between_valid_drift_and_land():
    valid = np.zeros((3, 5), dtype=bool)
    valid[:, 0] = True
    land = np.zeros((3, 5), dtype=bool)
    land[:, -1] = True

    taper = _coastal_drift_taper(valid, land)

    np.testing.assert_allclose(
        taper[1], [1.0, 0.75, 0.5, 0.25, 0.0])


def test_coastal_drift_taper_leaves_domain_unchanged_without_land():
    valid = np.zeros((3, 5), dtype=bool)
    valid[:, 0] = True

    taper = _coastal_drift_taper(valid, np.zeros_like(valid))

    np.testing.assert_array_equal(taper, np.ones_like(taper))
