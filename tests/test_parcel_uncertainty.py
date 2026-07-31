import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from driftaware_sialt.stacking.drift_aware_processor import (
    DriftAwareProcessor,
    grid_mission_fractions,
    grid_parcel_uncertainty,
    ocean_heat_flux_at_points,
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


def test_grid_mission_fractions_allows_missing_configured_sensor():
    data = gpd.GeoDataFrame(
        {
            "cryosat2": [1.0, 0.0],
            "sentinel3a": [0.0, 1.0],
        },
        geometry=[Point(-0.5, 0), Point(0.5, 0)],
        crs="EPSG:6931")
    grid = gpd.GeoDataFrame(
        geometry=[box(-1, -1, 1, 1)],
        crs=data.crs)

    mission_grid = grid_mission_fractions(
        data,
        grid,
        ["cryosat2", "sentinel3a", "sentinel3b"],
        ["cryosat2", "sentinel3a"])

    assert mission_grid.iloc[0]["cryosat2"] == pytest.approx(0.5)
    assert mission_grid.iloc[0]["sentinel3a"] == pytest.approx(0.5)
    assert mission_grid.iloc[0]["sentinel3b"] == 0.0
    assert mission_grid.iloc[0]["sentinel3b_sum"] == 0.0


def test_constant_ocean_heat_flux_is_applied_to_every_parcel():
    values = ocean_heat_flux_at_points(
        2.0,
        [100.0, 200.0, 300.0],
        [400.0, 500.0, 600.0])

    assert values.tolist() == [2.0, 2.0, 2.0]


def test_reanalysis_ocean_heat_flux_is_interpolated_at_parcels():
    class HeatFluxProduct:
        def interp_ocean_heat_flux(self, x, y):
            return np.asarray(x) + np.asarray(y)

    values = ocean_heat_flux_at_points(
        HeatFluxProduct(),
        [1.0, 2.0],
        [10.0, 20.0])

    assert values.tolist() == [11.0, 22.0]


def test_drift_correction_passes_constant_ocean_heat_flux_to_model():
    class Parent:
        sensor = ["cryosat2"]
        target_var = "sea_ice_thickness"
        add_variable = []
        out_epsg = "EPSG:6931"

    class DriftProduct:
        product_id = "drift"
        ice_drift = {"time_bnds": np.array([0.0])}

        @staticmethod
        def drift_correction(x, y):
            zeros = np.zeros(len(x))
            return np.full(len(x), 10.0), zeros, zeros

        @staticmethod
        def deformation(x, y):
            return np.zeros(len(x)), np.zeros(len(x))

    class ConcentrationProduct:
        product_id = "concentration"
        ice_conc_ahead = object()

        @staticmethod
        def interp_ice_concentration(dataset, x, y):
            return np.ones(len(x))

    class TemperatureProduct:
        received_heat_flux = None

        def thermodyn_growth(
                self, model, thickness, snow_depth, x, y, direction,
                heat_flux):
            self.received_heat_flux = heat_flux.to_numpy()
            return (
                np.full(len(x), -10.0),
                np.full(len(x), 0.1),
                np.full(len(x), 1.1),
            )

    processor = DriftAwareProcessor(Parent())
    processor.i = 0
    temperature = TemperatureProduct()
    parcels = pd.DataFrame({
        "xu": [0.0],
        "yu": [0.0],
        "dt_days": [0],
        "t0": [12 * 60 * 60],
        "geometry": [[Point(0, 0)]],
        "sea_ice_thickness_drift_unc": [0.0],
        "sit_corr_thermo_mod": [1.0],
        "snow_depth": [0.2],
        "thermo_growth_mod": [0.0],
        "thermo_change_mod": [0.0],
        "divergence": [[0.0]],
        "shear": [[0.0]],
        "_ice_conc_products": ["concentration"],
        "_ice_drift_products": [""],
    })

    result = processor.apply_drift_correction(
        1,
        parcels,
        DriftProduct(),
        ConcentrationProduct(),
        temperature,
        2.0,
        1,
        thermo_model="winter_2layers")

    assert temperature.received_heat_flux.tolist() == [2.0]
    assert result["ohf"].tolist() == [2.0]
    assert result["xu"].tolist() == [120.0]
