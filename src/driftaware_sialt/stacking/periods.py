"""Plan output dates and supporting observations without touching the filesystem."""

import datetime as dt
from dataclasses import dataclass
import re


DAY = dt.timedelta(days=1)


@dataclass(frozen=True)
class RunPeriod:
    output_start: dt.date
    output_end: dt.date
    processing_start: dt.date
    processing_end: dt.date

    @property
    def start(self):
        return dt.datetime.combine(self.processing_start, dt.time())

    @property
    def length(self):
        return (self.processing_end - self.processing_start).days + 1

    @property
    def output_days(self):
        offset = (self.output_start - self.processing_start).days
        return range(offset, offset + (self.output_end - self.output_start).days + 1)

    @property
    def label(self):
        return f'{self.output_start:%Y%m%d}-{self.output_end:%Y%m%d}'

    def to_dict(self):
        return {name: getattr(self, name).isoformat() for name in self.__dataclass_fields__}

    @classmethod
    def from_config(cls, config):
        try:
            period = cls(**{
                name: dt.date.fromisoformat(config['run'][name])
                for name in cls.__dataclass_fields__})
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('run requires planned output and processing dates') from error
        if not (period.processing_start <= period.output_start <= period.output_end <= period.processing_end):
            raise ValueError('invalid output/processing date order')
        return period


def plan_periods(hemisphere, window, mode, *, years=None, start=None, end=None):
    """Expand years or explicit inclusive dates; pad only SH annual runs."""
    if hemisphere not in ('nh', 'sh'):
        raise ValueError('hemisphere must be nh or sh')
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError('stacking.t_window must be a positive integer')
    if mode not in ('f', 'r', 'fr'):
        raise ValueError('stacking.mode must be f, r, or fr')
    if years is not None:
        if start is not None or end is not None:
            raise ValueError('--years cannot be combined with --start/--end')
        if not re.fullmatch(r'\d{4}(?::\d{4})?', str(years)):
            raise ValueError('--years must be YYYY or YYYY:YYYY (inclusive)')
        bounds = [int(value) for value in str(years).split(':')]
        first, last = bounds[0], bounds[-1]
        if not 1 <= first <= last <= 9998:
            raise ValueError('invalid or reversed year range')
        dates = [
            (dt.date(year, 10, 1), dt.date(year + 1, 4, 30))
            if hemisphere == 'nh' else (dt.date(year, 1, 1), dt.date(year, 12, 31))
            for year in range(first, last + 1)]
    else:
        if start is None or end is None:
            raise ValueError('provide --years or both --start and --end')
        try:
            dates = [(dt.date.fromisoformat(str(start)), dt.date.fromisoformat(str(end)))]
        except ValueError as error:
            raise ValueError('dates must use YYYY-MM-DD') from error
        if dates[0][0] > dates[0][1]:
            raise ValueError('--end must not precede --start')

    padding = window - 1 if years is not None and hemisphere == 'sh' else 0
    return [RunPeriod(
        begin, finish,
        begin - padding * DAY if 'f' in mode else begin,
        finish + padding * DAY if 'r' in mode else finish,
    ) for begin, finish in dates]
