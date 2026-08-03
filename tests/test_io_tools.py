import geopandas as gpd
from shapely.geometry import MultiPoint

from driftaware_sialt.io_tools import read_dasit_csv, write_dasit_csv


def test_trajectory_csv_uses_reduced_precision_without_mutating_input(tmp_path):
    geometry = MultiPoint([(1.2345, 2.6789), (3.8765, 4.1234)])
    data = gpd.GeoDataFrame(
        {
            'sea_ice_thickness': [1.23456789],
            'cryosat2': [0.66666667],
            'cryosat2_cnt': [3.0],
            'sentinel3a_cnt': [0.0],
        },
        geometry=[geometry],
        crs='EPSG:6931')
    config = {
        'options': {
            'out_epsg': 'EPSG:6931',
            'target_variable': 'sea_ice_thickness',
        },
        'stacking': {
            'mode': 'fr',
            't_window': 16,
            'hist': {
                'n_bins': 40,
                'range': {'freeboard': [0, 1], 'thickness': [0, 10]},
            },
        },
    }
    output_file = tmp_path / 'trajectory.csv'

    write_dasit_csv(data, output_file, config)
    result, metadata = read_dasit_csv(output_file, return_metadata=True)

    assert [(point.x, point.y)
            for point in result.geometry.iloc[0].geoms] == [
                (1.0, 3.0), (4.0, 4.0)]
    assert result['sea_ice_thickness'].iloc[0] == 1.2346
    assert result['cryosat2'].iloc[0] == 0.6667
    assert result['cryosat2_cnt'].iloc[0] == 3
    assert result['sentinel3a_cnt'].iloc[0] == 0
    assert metadata['crs'] == 'EPSG:6931'

    assert [(point.x, point.y)
            for point in data.geometry.iloc[0].geoms] == [
                (1.2345, 2.6789), (3.8765, 4.1234)]
    assert data['sea_ice_thickness'].iloc[0] == 1.23456789

    text = output_file.read_text()
    assert 'MULTIPOINT (1 3, 4 4)' in text
    assert '1.23456789' not in text
