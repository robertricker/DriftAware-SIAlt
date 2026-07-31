import datetime
import os
import re
from pathlib import Path

import pandas as pd
import xarray as xr
from loguru import logger

from driftaware_sialt.products.sea_ice_concentration import (
    SeaIceConcentrationProducts,
)
from driftaware_sialt.products.selection import select_product
from driftaware_sialt.volume.calculations import calculate_volume_and_mass
from driftaware_sialt.volume.output import (
    build_regional_summary,
    write_volume_dataset,
)


VOLUME_VARIABLES = {
    'sea_ice_volume',
    'sea_ice_mass',
    'snow_volume',
    'snow_mass',
    'sea_ice_extent',
    'sea_ice_area',
    'sea_ice_extent_total',
    'sea_ice_area_total',
    'sea_ice_volume_total',
    'sea_ice_mass_total',
}
CONTEXT_VARIABLES = {
    'crs',
    'region_code',
    'time_bnds',
    'longitude',
    'latitude',
}


def _resolve_subdirectory(root, configured):
    path = Path(configured)
    return path if path.is_absolute() else Path(root) / path


def _make_volume_filename(source_filename):
    parts = Path(source_filename).stem.split('-')
    try:
        product_level_index = parts.index('L3C')
    except ValueError as error:
        raise ValueError(
            f'gridded filename has no L3C product level: '
            f'{source_filename}') from error
    if len(parts) <= product_level_index + 2:
        raise ValueError(
            f'unexpected gridded filename layout: {source_filename}')
    parts[product_level_index] = 'L4'
    parts[product_level_index + 2] = 'SIVOL'
    return '-'.join(parts) + '.nc'


def _select_volume_product_variables(dataset):
    keep = [
        name for name in dataset.variables
        if name in VOLUME_VARIABLES or name in CONTEXT_VARIABLES
    ]
    result = dataset[keep]
    result.attrs.update({
        'title': 'Gridded sea ice volume and mass',
        'summary': (
            'Sea ice and snow volume and mass derived from gridded sea ice '
            'thickness and concentration'),
        'processing_level': 'L4',
        'production_date': datetime.datetime.utcnow().strftime(
            '%Y%m%dT%H%M%SZ'),
    })
    return result


def run(config):
    hem = config["options"]["hemisphere"]
    out_epsg = config["options"]["out_epsg"]
    volume_opt = config['volume']
    which_ice_conc = volume_opt.get('ice_conc_product')
    target_var = config['options']['target_variable']
    if target_var != 'sea_ice_thickness':
        raise ValueError(
            'volume command requires options.target_variable: '
            'sea_ice_thickness')

    gridded_data_dir = volume_opt.get('gridded_data_dir')
    if not gridded_data_dir:
        raise ValueError(
            'volume.gridded_data_dir must select a gridded run directory')
    input_dir = _resolve_subdirectory(
        config['output_dir']['gridded_data'], gridded_data_dir)
    file_list = sorted(input_dir.rglob('*.nc'))
    if not file_list:
        raise FileNotFoundError(
            f'no gridded NetCDF files found for volume computation in '
            f'{input_dir}')

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = (
        Path(config['output_dir']['volume'])
        / f'sea_ice_volume-{hem}-{timestamp}'
    )
    out_dir.mkdir(parents=True)

    si_density_param = volume_opt['sea_ice_density']
    snow_density_param = volume_opt['snow_density']
    interp_missing_sit = volume_opt.get('interp_missing_sit', True)
    sic_interp_threshold = volume_opt.get('sic_interp_threshold', 15)

    sic_products = None
    if which_ice_conc:
        if which_ice_conc not in config['auxiliary']['ice_conc']:
            raise KeyError(
                f'no auxiliary.ice_conc directory configured for '
                f'{which_ice_conc}')
        sic_products = SeaIceConcentrationProducts.load_products(
            [which_ice_conc],
            config['auxiliary']['ice_conc'],
            hem=hem,
            out_epsg=out_epsg)

    summaries = []

    for file in file_list:
        time_str = re.search(r'-(\d{8})-', os.path.basename(file)).group(1)
        dt1d = datetime.timedelta(days=1)
        t0 = datetime.datetime.strptime(time_str, '%Y%m%d')
        t1 = t0 + dt1d

        with xr.open_dataset(file, decode_times=False) as source:
            data = source.load()

        if which_ice_conc:
            sic_product = select_product(sic_products, t0, t1)
            if not sic_product:
                logger.warning(f"{t0:%Y%m%d}: no nearby concentration file; skipping")
                continue
            ice_conc = sic_product.get_ice_concentration(sic_product.target_files)
        else:
            if 'sea_ice_concentration' not in data:
                raise KeyError(
                    f'{file} does not contain sea_ice_concentration; configure '
                    f'volume.ice_conc_product to use an external product')
            ice_conc = data['sea_ice_concentration']

        dataset = calculate_volume_and_mass(
            data,
            ice_conc,
            target_var,
            si_density_param,
            snow_density_param,
            interp_missing_sit=interp_missing_sit,
            sic_threshold=sic_interp_threshold
        )
        dataset = _select_volume_product_variables(dataset)

        dated_out_dir = out_dir / time_str[:4] / time_str[4:6]
        outfile = dated_out_dir / _make_volume_filename(file.name)
        write_volume_dataset(dataset, str(outfile))
        summaries.append(build_regional_summary(dataset))

    if not summaries:
        raise RuntimeError('no volume files were generated')

    df_vol_mass = pd.concat(summaries, axis=0)
    df_vol_mass['time'] = pd.to_datetime(
        df_vol_mass.index, unit='s', origin='unix')
    df_vol_mass.set_index('time', inplace=True)

    first_date = df_vol_mass.index.min().strftime('%Y%m%d')
    last_date = df_vol_mass.index.max().strftime('%Y%m%d')
    summary_name = re.sub(
        r'-\d{8}-',
        f'-{first_date}-{last_date}-',
        _make_volume_filename(file_list[0].name),
        count=1,
    ).replace('.nc', '.csv')
    outfile_csv = out_dir / summary_name
    df_vol_mass.to_csv(outfile_csv)
    logger.info('volume summary saved as: {}', outfile_csv)
