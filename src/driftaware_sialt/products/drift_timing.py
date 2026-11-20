"""Native OSI-SAF drift intervals, expressed on a common UTC clock."""

import datetime as dt
from pathlib import Path
import re

import netCDF4
import numpy as np


DRIFT_PRODUCTS = ('osi455', 'osi435', 'osi405')
EPOCH = dt.datetime(1970, 1, 1)
EPOCH_UNITS = 'seconds since 1970-01-01 00:00:00'
DAY = dt.timedelta(days=1)


def filename_interval(filename, product_id):
    """Index native intervals cheaply; the selected file's metadata is authoritative."""
    stamps = re.findall(r'\d{12}', Path(filename).name)
    if product_id not in DRIFT_PRODUCTS or len(stamps) != (2 if product_id == 'osi405' else 1):
        raise ValueError(f'unrecognised {product_id} drift filename: {filename}')
    times = [dt.datetime.strptime(value, '%Y%m%d%H%M') for value in stamps]
    start, end = times if product_id == 'osi405' else (times[0] - DAY, times[0])
    if start >= end:
        raise ValueError(f'invalid drift interval: {filename}')
    return start, end


def epoch_values(variable, units=None, calendar='standard'):
    units = getattr(variable, 'units', units)
    calendar = getattr(variable, 'calendar', calendar)
    if not units or calendar not in ('standard', 'gregorian', 'proleptic_gregorian'):
        raise ValueError('drift timestamps require CF time units and a Gregorian calendar')
    # CF seconds/minutes/hours/days are affine; avoid decoding every grid cell.
    reference = netCDF4.num2date([0., 1.], units, calendar=calendar)
    offset, next_value = netCDF4.date2num(reference, EPOCH_UNITS, calendar=calendar)
    return np.ma.asarray(variable[:], dtype=float) * (next_value - offset) + offset


def time_bounds(data):
    variable = data.variables['time_bnds']
    time = data.variables.get('time')
    bounds = epoch_values(variable, getattr(time, 'units', None),
                          getattr(time, 'calendar', 'standard')).ravel()
    if bounds.size != 2 or np.ma.getmaskarray(bounds).any() or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
        raise ValueError('drift time_bnds must contain two finite, increasing timestamps')
    return np.asarray(bounds)


def vector_duration(data, bounds):
    """Use per-vector observation times when supplied, otherwise native bounds."""
    if 't0' in data.variables and 't1' in data.variables:
        units = getattr(data.variables['time_bnds'], 'units',
                        getattr(data.variables.get('time'), 'units', None))
        calendar = getattr(data.variables['time_bnds'], 'calendar',
                           getattr(data.variables.get('time'), 'calendar', 'standard'))
        return epoch_values(data['t1'], units, calendar) - epoch_values(data['t0'], units, calendar)
    start, end = bounds
    factors = {'seconds': 1., 's': 1., 'minutes': 60., 'hours': 3600., 'days': 86400.}
    offsets = []
    for name in ('dt0', 'dt1'):
        if name not in data.variables:
            offsets.append(0.)
        else:
            variable = data[name]
            if getattr(variable, 'units', None) not in factors:
                raise ValueError(f'unsupported drift {name} units')
            offsets.append(np.ma.asarray(variable[:]) * factors[variable.units])
    return end - start + offsets[1] - offsets[0]
