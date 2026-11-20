import datetime as dt

import pytest

from driftaware_sialt.input_data import required_input_dates
from driftaware_sialt.stacking.periods import RunPeriod, plan_periods


@pytest.mark.parametrize('year, days', [(2018, 365), (2020, 366)])
@pytest.mark.parametrize('window', [1, 5, 16])
@pytest.mark.parametrize('mode', ['f', 'r', 'fr'])
def test_sh_year_padding_and_sync_dates(year, days, window, mode):
    period = plan_periods('sh', window, mode, years=str(year))[0]
    padding = dt.timedelta(days=window - 1)
    first, last = dt.date(year, 1, 1), dt.date(year, 12, 31)
    assert period.output_start == first
    assert period.output_end == last
    assert period.processing_start == (first - padding if 'f' in mode else first)
    assert period.processing_end == (last + padding if 'r' in mode else last)
    assert len(period.output_days) == days
    assert RunPeriod.from_config({'run': period.to_dict()}) == period
    dates = required_input_dates({'run': period.to_dict(), 'stacking': {'mode': mode}})
    assert len(dates['altimetry']) == period.length
    assert dates['altimetry'][0].date() == period.processing_start
    assert dates['altimetry'][-1].date() == period.processing_end


@pytest.mark.parametrize('years, first, last', [
    ('2020:2024', 2020, 2024),
    ('2020:2021', 2020, 2021),
    ('2020', 2020, 2021),
])
def test_nh_years_are_unpadded_october_to_april_winters(years, first, last):
    periods = plan_periods('nh', 16, 'fr', years=years)
    assert [(p.output_start, p.output_end) for p in periods] == [
        (dt.date(year, 10, 1), dt.date(year + 1, 4, 30))
        for year in range(first, last)]
    assert all(p.output_start == p.processing_start and p.output_end == p.processing_end for p in periods)


def test_nh_same_year_range_is_rejected():
    with pytest.raises(ValueError, match='use YYYY for one winter'):
        plan_periods('nh', 16, 'fr', years='2020:2020')


@pytest.mark.parametrize('hemisphere', ['nh', 'sh'])
def test_explicit_dates_are_one_continuous_unpadded_run(hemisphere):
    period, = plan_periods(hemisphere, 16, 'fr', start='2018-03-01', end='2020-09-30')
    assert period.processing_start == period.output_start == dt.date(2018, 3, 1)
    assert period.processing_end == period.output_end == dt.date(2020, 9, 30)


@pytest.mark.parametrize('arguments', [
    {}, {'start': '2018-01-01'}, {'end': '2018-01-01'},
    {'years': '2018:2017'}, {'years': '2018-2024'}, {'years': '0000'},
    {'years': '2018', 'end': '2018-01-01'},
    {'start': '2020-02-30', 'end': '2020-03-01'},
    {'start': '2020-03-02', 'end': '2020-03-01'},
])
def test_invalid_date_selection_is_rejected(arguments):
    with pytest.raises(ValueError):
        plan_periods('sh', 5, 'fr', **arguments)


@pytest.mark.parametrize('window', [0, -1, True, 2.5])
def test_invalid_window_is_rejected(window):
    with pytest.raises(ValueError):
        plan_periods('sh', window, 'fr', years='2018')


def test_partial_days_cannot_appear_in_snapshot_dates():
    with pytest.raises(ValueError):
        RunPeriod.from_config({'run': {'output_start': '2018-01-01 12:00:00'}})
