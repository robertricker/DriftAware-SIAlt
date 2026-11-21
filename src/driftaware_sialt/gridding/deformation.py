"""Time-weighted strain-rate means over the sampled advection interval."""

import datetime as dt

import numpy as np
import pandas as pd


RATE_FIELDS = ('divergence', 'shear', 'deformation')


def mean_trajectory_rates(data, reference_time):
    """Average actual samples, excluding initialization and zero-lag histories.

    The first step runs from the acquisition timestamp to an adjacent day's
    noon. All subsequent steps last one day. Positive elapsed durations weight
    the physical rates in both directions; backward tracking does not flip
    divergence. A rate requires a complete finite history; shear must be
    nonnegative. Missing rates do not invalidate the thickness observation.
    """
    result = pd.DataFrame(np.nan, index=data.index, columns=RATE_FIELDS)
    reference = (reference_time - dt.datetime(1970, 1, 1)).total_seconds()
    lags = data['dt_days'].to_numpy(dtype=float)
    if not np.isfinite(lags).all() or np.any(lags != np.trunc(lags)):
        raise ValueError('strain-rate means require integer trajectory acquisition-day lags')

    for lag, group in data.groupby(data['dt_days'].abs()):
        steps = int(lag)
        histories = []
        for name in ('divergence', 'shear'):
            try:
                values = np.asarray(group[name].tolist(), dtype=float)
            except ValueError as error:
                raise ValueError(f'inconsistent trajectory {name} histories') from error
            if values.shape != (len(group), steps + 1) or np.any(values[:, 0] != 0):
                raise ValueError(f'trajectory {name} history must contain an initial zero and {steps} steps')
            histories.append(values[:, 1:])
        if steps == 0:
            continue

        direction = -np.sign(group['dt_days'].to_numpy(dtype=float))
        duration = direction * (reference - group['t0'].to_numpy(dtype=float)) / 86400.
        first_step = duration - (steps - 1)
        if not np.isfinite(duration).all() or np.any(first_step <= 0) or np.any(first_step > 2):
            raise ValueError('trajectory acquisition time is inconsistent with its lag and reference noon')
        durations = np.ones_like(histories[0])
        durations[:, 0] = first_step
        divergence, shear = histories
        shear = np.where(shear >= 0, shear, np.nan)
        for name, rates in zip(RATE_FIELDS, (divergence, shear, np.hypot(divergence, shear))):
            valid = np.isfinite(rates).all(axis=1)
            result.loc[group.index[valid], name] = (
                np.sum(rates[valid] * durations[valid], axis=1) / duration[valid])
    return result
