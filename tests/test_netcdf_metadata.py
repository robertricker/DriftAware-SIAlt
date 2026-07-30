from pathlib import Path

import xarray as xr
import yaml
from jinja2 import Template

from driftaware_sialt.gridding.prepare_netcdf import PrepareNetcdf
from driftaware_sialt.volume.output import load_variable_attributes


ROOT = Path(__file__).parents[1]
GRIDDING_CONFIG = (
    ROOT / "src/driftaware_sialt/gridding/netcdf_config.yaml")
VOLUME_CONFIG = ROOT / "src/driftaware_sialt/volume/netcdf_config.yaml"

CF_STANDARD_NAMES_USED = {
    "air_temperature",
    "divergence_of_sea_ice_velocity",
    "latitude",
    "longitude",
    "magnitude_of_sea_ice_displacement",
    "projection_x_coordinate",
    "projection_y_coordinate",
    "sea_ice_area",
    "sea_ice_area_fraction",
    "sea_ice_extent",
    "sea_ice_freeboard",
    "sea_ice_mass",
    "sea_ice_thickness",
    "sea_ice_volume",
    "surface_snow_thickness",
    "tendency_of_sea_ice_thickness_due_to_dynamics",
    "tendency_of_sea_ice_thickness_due_to_thermodynamics",
    "time",
    "upward_sea_ice_basal_heat_flux",
}
CF_STANDARD_NAME_MODIFIERS_USED = {"standard_error"}


def load_config(path):
    with path.open() as stream:
        return yaml.safe_load(stream)


def test_all_configured_variables_have_descriptive_metadata():
    for path in (GRIDDING_CONFIG, VOLUME_CONFIG):
        config = load_config(path)
        for mode, mapping in config["variables"].items():
            assert len(mapping["include"]) == len(mapping["rename"]), (
                path, mode)

        attributes = config["variable_attributes"]
        for variable_name, metadata in attributes.items():
            assert metadata.get("long_name"), (path, variable_name, "long_name")
            assert metadata.get("units"), (path, variable_name, "units")


def test_all_possible_thickness_outputs_have_metadata():
    config = load_config(GRIDDING_CONFIG)
    output_variables = {
        "hist_bins",
        "latitude",
        "longitude",
        "sea_ice_thickness_hist",
        "sea_ice_thickness_std",
        "time",
        "time_bnds",
        "xc",
        "yc",
    }
    for mode in config["variables"].values():
        output_variables.update(
            Template(name).render(target_var="sea_ice_thickness")
            for name in mode["rename"])

    attributes = config["variable_attributes"]
    assert output_variables <= attributes.keys()


def test_standard_names_are_verified_cf_names():
    for path in (GRIDDING_CONFIG, VOLUME_CONFIG):
        attributes = load_config(path)["variable_attributes"]
        for variable_name, metadata in attributes.items():
            if "standard_name" not in metadata:
                continue
            parts = metadata["standard_name"].split()
            assert parts[0] in CF_STANDARD_NAMES_USED, (
                path, variable_name, parts[0])
            assert set(parts[1:]) <= CF_STANDARD_NAME_MODIFIERS_USED, (
                path, variable_name, parts[1:])


def test_volume_calculation_units():
    attributes = load_config(VOLUME_CONFIG)["variable_attributes"]
    expected_units = {
        "sea_ice_area": "m2",
        "sea_ice_area_total": "m2",
        "sea_ice_extent": "m2",
        "sea_ice_extent_total": "m2",
        "sea_ice_mass": "kg",
        "sea_ice_mass_total": "kg",
        "sea_ice_volume": "m3",
        "sea_ice_volume_total": "m3",
        "snow_mass": "kg",
        "snow_volume": "m3",
    }
    assert {
        name: attributes[name]["units"] for name in expected_units
    } == expected_units


def test_ancillary_variables_only_reference_present_variables():
    preparer = PrepareNetcdf.__new__(PrepareNetcdf)
    preparer.netcdf_config = load_config(GRIDDING_CONFIG)
    dataset = xr.Dataset(
        {
            "sea_ice_thickness": ("x", [1.0]),
            "sea_ice_thickness_parcel_unc": ("x", [0.1]),
        })

    result = preparer.set_var_attrbs(dataset)

    assert (
        result["sea_ice_thickness"].attrs["ancillary_variables"]
        == "sea_ice_thickness_parcel_unc")


def test_volume_metadata_includes_shared_gridding_variables():
    attributes = load_variable_attributes()

    assert "sea_ice_thickness_dynamic_tendency" in attributes
    assert "model_sea_ice_thickness" in attributes
    assert "sentinel3a_frac" in attributes
