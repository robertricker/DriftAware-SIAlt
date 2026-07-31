import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from driftaware_sialt.gridding.gridding_lib import grid_data


def test_count_weighting_uses_available_missions_per_grid_cell():
    grid = gpd.GeoDataFrame(
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs='EPSG:6931')
    data = gpd.GeoDataFrame(
        {
            'value': [2.0, 4.0, 2.0, 4.0, 8.0],
            'cryosat2_cnt': [16.0, 0.0, 16.0, 0.0, 0.0],
            'sentinel3a_cnt': [0.0, 4.0, 0.0, 16.0, 0.0],
            'sentinel3b_cnt': [0.0, 0.0, 0.0, 0.0, 16.0],
        },
        geometry=[Point(2, 2), Point(4, 4),
                  Point(12, 2), Point(14, 4), Point(16, 6)],
        crs=grid.crs)

    result = grid_data(
        data,
        grid,
        ['value', 'cryosat2_cnt', 'sentinel3a_cnt', 'sentinel3b_cnt'],
        ['value', 'cryosat2_cnt', 'sentinel3a_cnt', 'sentinel3b_cnt'],
        agg_mode=['weighted_mean'],
        weight_var='counts')

    assert result.iloc[0]['value'] == pytest.approx(8.0 / 3.0)
    assert result.iloc[1]['value'] == pytest.approx(4.0)


@pytest.mark.parametrize('weight_var', ['dt_days', ['dt_days'], None])
def test_dt_days_weighting_is_symmetric_and_favours_day_zero(weight_var):
    grid = gpd.GeoDataFrame(
        geometry=[box(0, 0, 10, 10)], crs='EPSG:6931')
    data = gpd.GeoDataFrame(
        {
            'value': [0.0, 10.0, 40.0],
            'dt_days': [-1.0, 0.0, 1.0],
        },
        geometry=[Point(2, 2), Point(4, 4), Point(6, 6)],
        crs=grid.crs)

    result = grid_data(
        data,
        grid,
        ['value', 'dt_days'],
        ['value', 'dt_days'],
        agg_mode=['weighted_mean'],
        weight_var=weight_var)

    # The weights for days -1, 0 and +1 are 0.25, 1 and 0.25.
    assert result.iloc[0]['value'] == pytest.approx(40.0 / 3.0)
