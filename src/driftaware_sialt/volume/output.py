"""NetCDF and tabular output for the volume-processing stage."""

import os
from pathlib import Path
import pandas as pd
import yaml
from loguru import logger


def load_variable_attributes():
    """Combine shared gridding metadata with volume-specific overrides."""
    metadata_paths = [
        Path(__file__).parents[1] / "gridding" / "netcdf_config.yaml",
        Path(__file__).with_name("netcdf_config.yaml"),
    ]
    variable_attributes = {}
    for metadata_path in metadata_paths:
        with metadata_path.open() as stream:
            variable_attributes.update(
                yaml.safe_load(stream)["variable_attributes"])
    return variable_attributes


def write_volume_dataset(dataset, outfile):
    # Source NetCDF encodings can duplicate attributes such as ``coordinates``.
    # Start the derived volume product with clean encodings.
    dataset = dataset.copy(deep=False)
    for variable in dataset.variables.values():
        variable.encoding = {}

    for variable_name, attributes in load_variable_attributes().items():
        if variable_name in dataset:
            dataset[variable_name].attrs.update(attributes)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    compression = {"zlib": True, "complevel": 1}
    encoding = {
        variable_name: compression for variable_name in dataset.data_vars
    }

    dataset.to_netcdf(outfile, encoding=encoding, format="NETCDF4")
    logger.info("Volumes and masses saved as : %s" % outfile)


def build_regional_summary(dataset):
    spatial_variables = [
        "snow_volume",
        "snow_mass",
        "sea_ice_volume",
        "sea_ice_mass",
        "sea_ice_extent",
        "sea_ice_area",
    ]
    total_variables = [
        "sea_ice_extent_total",
        "sea_ice_area_total",
        "sea_ice_volume_total",
        "sea_ice_mass_total",
    ]
    spatial_variables = [
        variable for variable in spatial_variables if variable in dataset
    ]
    total_variables = [
        variable for variable in total_variables if variable in dataset
    ]

    spatial_dataset = dataset[spatial_variables]
    summary_dataset = spatial_dataset.sum(dim=["xc", "yc"])
    for variable in total_variables:
        summary_dataset[variable] = dataset[variable]
    summary = summary_dataset.to_dataframe()
    region_codes = dataset.region_code.attrs["flag_values"]

    for region in region_codes:
        region_summary = (
            spatial_dataset
            .where(dataset.region_code == region)
            .sum(dim=["xc", "yc"])
            .to_dataframe()
        )
        region_summary = region_summary.rename(
            columns={
                variable: f"{variable}_{region}"
                for variable in region_summary.columns
            }
        )
        summary = pd.concat([summary, region_summary], axis=1)

    return summary
