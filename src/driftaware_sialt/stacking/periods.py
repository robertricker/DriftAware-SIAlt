"""Date ranges shared by stacking and input synchronization."""

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class YearStackingPeriod:
    start: dt.datetime
    length: int
    output_days: range


def year_stacking_period(start, window, hemisphere):
    """Pad SH processing around outputs from start through 31 December.

    ``output_days`` indexes the target dates within the padded processing range.
    The stacking window includes lag zero, hence each buffer has window - 1 days.
    """
    if hemisphere != "sh":
        raise ValueError("stacking.t_length='year' requires hemisphere 'sh'")
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("stacking.t_window must be a positive integer")

    end = start.replace(year=start.year + 1, month=1, day=1)
    output_length = (end - start).days
    padding = window - 1
    return YearStackingPeriod(
        start=start - dt.timedelta(days=padding),
        length=output_length + 2 * padding,
        output_days=range(padding, padding + output_length),
    )
