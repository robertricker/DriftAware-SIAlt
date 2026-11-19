import datetime as dt
from urllib.error import HTTPError

import numpy as np
from PIL import Image
from pyproj import CRS
import pytest
import xarray as xr

from driftaware_sialt.visualization import sar


def _daily_image(tmp_path, day, pixels):
    path = tmp_path / f'{day}.vrt'
    # Lossless tiny fixtures isolate geolocation and date selection from JPEG noise.
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(path.with_suffix('.jpg'), format='PNG')
    path.write_text(
        '<VRTDataset rasterXSize="2" rasterYSize="2">'
        f'<SRS>{CRS.from_epsg(6931).to_wkt()}</SRS>'
        '<GeoTransform>0,1000,0,2000,0,-1000</GeoTransform>'
        '<VRTRasterBand><ComplexSource><LUT>0:0,15:0,15:15,255:255</LUT>'
        '</ComplexSource></VRTRasterBand></VRTDataset>')
    return path, f'http://example.test/{day}'


def test_sar_preserves_closest_dates_and_pixel_locations(tmp_path, monkeypatch):
    day = dt.date(2018, 12, 1)
    inputs = {
        0: [[50, 0], [0, 0]], -1: [[90, 60], [0, 0]],
        1: [[150, 160], [170, 0]], -2: [[180, 180], [180, 80]],
        2: [[220, 220], [220, 220]],
    }
    monkeypatch.setattr(sar, 'download_dtu_day', lambda d, cache: _daily_image(
        tmp_path, d, inputs[(d - day).days]))
    result = sar.build_sar_mosaic(day, tmp_path, days=2, resolution=1000, extent=[0, 2000, 0, 2000])
    np.testing.assert_array_equal(result.sar_brightness, [[50, 60], [170, 80]])
    np.testing.assert_array_equal(result.source_day_offset, [[0, -1], [1, -2]])
    np.testing.assert_array_equal(result.xc, [500, 1500])
    np.testing.assert_array_equal(result.yc, [1500, 500])
    assert CRS.from_wkt(result.crs.attrs['spatial_ref']).to_epsg() == 6931
    assert result.attrs['window_start'] == '2018-11-29'
    assert 'not calibrated backscatter' in result.attrs['limitations']


def test_sar_leaves_gaps_empty_and_records_missing_days(tmp_path, monkeypatch):
    day = dt.date(2018, 12, 1)

    def download(d, cache):
        if d != day:
            raise HTTPError('http://example.test', 404, 'not found', None, None)
        return _daily_image(tmp_path, d, [[50, 0], [10, 15]])

    monkeypatch.setattr(sar, 'download_dtu_day', download)
    result = sar.build_sar_mosaic(day, tmp_path, days=1, resolution=1000, extent=[0, 2000, 0, 2000])
    assert np.count_nonzero(result.sar_brightness) == 1
    assert result.attrs['unavailable_dates'] == '2018-11-30,2018-12-02'
    path = tmp_path / 'mosaic.nc'
    result.to_netcdf(path)
    with xr.open_dataset(path, decode_timedelta=False) as loaded:
        assert loaded.sar_brightness.count().item() == 1
        assert loaded.source_day_offset.count().item() == 1


def test_sar_rejects_unavailable_coverage(tmp_path, monkeypatch):
    def missing(d, cache):
        raise HTTPError('http://example.test', 404, 'not found', None, None)

    monkeypatch.setattr(sar, 'download_dtu_day', missing)
    with pytest.raises(ValueError, match='no SAR coverage'):
        sar.build_sar_mosaic('2018-12-01', tmp_path, days=0, extent=[0, 2000, 0, 2000])


def test_sar_does_not_hide_server_failures(tmp_path, monkeypatch):
    def failed(d, cache):
        raise HTTPError('http://example.test', 503, 'unavailable', None, None)

    monkeypatch.setattr(sar, 'download_dtu_day', failed)
    with pytest.raises(HTTPError):
        sar.build_sar_mosaic('2018-12-01', tmp_path, days=0, extent=[0, 2000, 0, 2000])
