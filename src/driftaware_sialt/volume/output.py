"""NetCDF and tabular output for the volume-processing stage."""

import os
import re
import shutil
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger


def organize_files_by_date(source_dir, target_dir):
    files = [name for name in os.listdir(source_dir) if name.endswith(".nc")]
    for filename in files:
        if os.path.basename(filename).startswith("._"):
            os.remove(os.path.join(source_dir, filename))
            continue
        match = re.search(r"(\d{8})", filename)
        if match:
            date_str = match.group(1)
            year, month = date_str[:4], date_str[4:6]
            month_dir = os.path.join(target_dir, year, month)
            os.makedirs(month_dir, exist_ok=True)
            shutil.move(
                os.path.join(source_dir, filename),
                os.path.join(month_dir, filename),
            )
        else:
            logger.info(f"Date not found in file name: {filename}")

    if not os.listdir(source_dir):
        os.rmdir(source_dir)
    else:
        logger.info(
            f"Source directory {source_dir} is not empty and has not been "
            "removed."
        )


def write_volume_dataset(dataset, outfile):
    with Path(__file__).with_name("netcdf_config.yaml").open() as stream:
        netcdf_config = yaml.safe_load(stream)

    for variable_name, attributes in netcdf_config[
            "variable_attributes"].items():
        if variable_name in dataset:
            dataset[variable_name].attrs.update(attributes)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    compression = {"zlib": True, "complevel": 1}
    encoding = {
        variable_name: compression for variable_name in dataset.data_vars
    }
    for variable_name in dataset.variables:
        dataset[variable_name].encoding = encoding.copy()

    dataset.to_netcdf(outfile, encoding=encoding, format="NETCDF4")
    logger.info("Volumes and masses saved as : %s" % outfile)


def build_regional_summary(dataset):
    summary_variables = [
        "snow_volume",
        "snow_mass",
        "sea_ice_volume",
        "sea_ice_mass",
        "sea_ice_extent",
        "sea_ice_area",
        "sea_ice_extent_total",
        "sea_ice_area_total",
        "sea_ice_volume_total",
        "sea_ice_mass_total",
    ]
    summary_variables = [
        variable for variable in summary_variables if variable in dataset
    ]

    summary_dataset = dataset[summary_variables]
    summary = summary_dataset.sum(dim=["xc", "yc"]).to_dataframe()
    region_codes = dataset.region_code.attrs["flag_values"]

    for region in region_codes:
        region_summary = (
            summary_dataset
            .where(dataset.region_code == region)
            .sum(dim=["xc", "yc"])
            .to_dataframe()
            / 1000000000
        )
        region_summary = region_summary.rename(
            columns={
                variable: f"{variable}_{region}"
                for variable in region_summary.columns
            }
        )
        summary = pd.concat([summary, region_summary], axis=1)

    return summary
