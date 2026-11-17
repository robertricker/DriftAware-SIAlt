from pathlib import Path

import pytest
import yaml

from driftaware_sialt.config_loader import _load_yaml


@pytest.fixture
def workflow_configs(tmp_path):
    """Real scientific defaults with private, tiny test repositories and grids."""
    root = Path(__file__).parents[1]
    common = _load_yaml(root / 'config/cci/common.yaml')
    common['base_dir'] = str(tmp_path)
    common['input_dir'] = {
        mission: f'input/{{hemisphere}}/{mission}'
        for mission in ('cryosat2', 'sentinel3a', 'sentinel3b')}
    paths = {}
    for stage in ('common', 'stacking', 'gridding'):
        if stage == 'common':
            config = common
        else:
            with (root / f'config/cci/{stage}.yaml').open() as stream:
                config = yaml.safe_load(stream)
            config[stage]['parallel'] = {'enabled': False, 'workers': 1}
            grid = 'parcel_grid' if stage == 'stacking' else 'netcdf_grid'
            config[stage][grid] = {'bounds': [-1, -1, 1, 1], 'dim': 1}
            if stage == 'stacking':
                config[stage]['growth_estimation']['land_area_correction']['enabled'] = False
        paths[stage] = tmp_path / f'{stage}.yaml'
        paths[stage].write_text(yaml.safe_dump(config, sort_keys=False))
    return paths
