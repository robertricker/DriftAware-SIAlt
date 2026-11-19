"""Coarse Arctic SAR display mosaics from DTU's daily Sentinel-1 images.

The JPEG values are display brightness, not calibrated radar backscatter.
Original VRT metadata defines the spherical source CRS and the no-data cutoff.
"""

import datetime as dt
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
from scipy.ndimage import map_coordinates
import xarray as xr

from driftaware_sialt.visualization.visualization_tools import _projection_and_extent


def download_dtu_day(date, cache_dir):
    """Cache one daily display image and its provider georeferencing unchanged."""
    directory = Path(cache_dir) / 'dtu' / f'{date:%Y%m%d}'
    directory.mkdir(parents=True, exist_ok=True)
    stem = f'{date:%Y%m%d}.s1ab.1km.n.mos'
    base = f'http://seaice.dk/N/{date:%Y/%m/%d}/{stem}'
    for suffix in ('vrt', 'info', 'jpg'):
        path = directory / f'{stem}.{suffix}'
        if not path.exists():
            with urlopen(f'{base}.{suffix}', timeout=60) as response:
                content = response.read()
            temporary = path.with_suffix(path.suffix + '.part')
            temporary.write_bytes(content)
            temporary.replace(path)
    return directory / f'{stem}.vrt', base


def build_sar_mosaic(date, cache_dir, *, days=2, resolution=2000, extent=None):
    """Return an EPSG:6931 display mosaic, preferring the closest daily image.

    Ties prefer the earlier day. Invalid pixels stay empty; no temporal average
    or ice-motion correction is applied. Offsets describe daily mosaic dates,
    not exact scene acquisition times. Resampling uses nearest source pixels.
    """
    date = dt.date.fromisoformat(str(date))
    if not isinstance(days, int) or isinstance(days, bool) or not 0 <= days <= 30:
        raise ValueError('days must be an integer between 0 and 30')
    if not np.isfinite(resolution) or resolution < 1000:
        raise ValueError('resolution must be at least 1000 metres')
    _, default_extent = _projection_and_extent('nh')
    xmin, xmax, ymin, ymax = default_extent if extent is None else extent
    if not (xmax > xmin and ymax > ymin):
        raise ValueError('extent must have increasing x and y bounds')
    x = np.arange(xmin + resolution / 2, xmax, resolution)
    y = np.arange(ymax - resolution / 2, ymin, -resolution)
    if not len(x) or not len(y):
        raise ValueError('extent must contain at least one output pixel')
    image = np.zeros((len(y), len(x)), dtype=np.uint8)
    offsets = np.full(image.shape, 127, dtype=np.int8)
    used, missing = [], []
    ordered = sorted(range(-days, days + 1), key=lambda offset: (abs(offset), offset))
    for offset in ordered:
        day = date + dt.timedelta(days=offset)
        try:
            vrt_path, url = download_dtu_day(day, cache_dir)
        except HTTPError as error:
            if error.code != 404:
                raise
            missing.append(day.isoformat())
            print('No archived daily mosaic:', day)
            continue
        vrt = ET.parse(vrt_path).getroot()
        source_crs = CRS.from_wkt(vrt.findtext('SRS'))
        transform = [float(value) for value in vrt.findtext('GeoTransform').split(',')]
        if transform[2] or transform[4] or transform[1] <= 0 or transform[5] >= 0:
            raise ValueError(f'unsupported source raster transform: {vrt_path}')
        with Image.open(vrt_path.with_suffix('.jpg')) as source:
            if source.mode != 'L':
                raise ValueError(f'expected a grayscale SAR display image: {vrt_path}')
            pixels = np.asarray(source).copy()
        if pixels.shape != (int(vrt.attrib['rasterYSize']), int(vrt.attrib['rasterXSize'])):
            raise ValueError(f'image dimensions disagree with VRT: {vrt_path}')
        # DTU's VRT LUT suppresses JPEG artifacts in the black no-data areas.
        lut = vrt.findtext('.//LUT')
        if lut != '0:0,15:0,15:15,255:255':
            raise ValueError(f'unrecognised display/no-data LUT: {vrt_path}')
        project = Transformer.from_crs('EPSG:6931', source_crs, always_xy=True)
        count = 0
        for start in range(0, len(y), 128):
            section = slice(start, start + 128)
            xx, yy = np.meshgrid(x, y[section])
            sx, sy = project.transform(xx, yy)
            col = (sx - transform[0]) / transform[1] - 0.5
            row = (sy - transform[3]) / transform[5] - 0.5
            sample = map_coordinates(pixels, [row, col], order=0,
                                     mode='constant', cval=0, prefilter=False)
            fill = (image[section] == 0) & (sample > 15)
            image[section][fill] = sample[fill]
            offsets[section][fill] = offset
            count += int(fill.sum())
        used.append(url)
        print(f'{day}: added {count:,} pixels')
    if not image.any():
        raise ValueError('no SAR coverage found in the requested dates and extent')
    result = xr.Dataset(
        {'sar_brightness': (('yc', 'xc'), image),
         'source_day_offset': (('yc', 'xc'), offsets),
         'crs': ((), 0)},
        coords={'xc': x, 'yc': y},
        attrs={'title': 'DTU Sentinel-1A/B Arctic SAR display mosaic',
               'target_date': date.isoformat(),
               'window_start': (date - dt.timedelta(days=days)).isoformat(),
               'window_end': (date + dt.timedelta(days=days)).isoformat(),
               'resolution_m': float(resolution), 'source_urls': '\n'.join(used),
               'unavailable_dates': ','.join(missing),
               'selection': 'Closest daily mosaic; earlier date wins ties; nearest-neighbour resampling.',
               'limitations': 'JPEG display brightness, not calibrated backscatter. No ice-motion correction. '
                              'Source dates refer to daily mosaics, not precise pixel acquisition times.',
               'credit': 'Copernicus Sentinel data, processed by DTU Space / seaice.dk.'})
    result.crs.attrs.update(CRS.from_epsg(6931).to_cf())
    result.crs.attrs['spatial_ref'] = CRS.from_epsg(6931).to_wkt()
    for name in ('sar_brightness', 'source_day_offset'):
        result[name].attrs['grid_mapping'] = 'crs'
    result.sar_brightness.attrs.update(long_name='SAR display brightness', units='1')
    result.source_day_offset.attrs.update(
        long_name='Source daily mosaic date minus target date (days)', units='1',
        comment='Integer day offset; negative values precede the target date.')
    result.sar_brightness.encoding.update(_FillValue=0, zlib=True, complevel=4)
    result.source_day_offset.encoding.update(_FillValue=127, zlib=True, complevel=4)
    result.xc.attrs.update(units='m', standard_name='projection_x_coordinate')
    result.yc.attrs.update(units='m', standard_name='projection_y_coordinate')
    return result
