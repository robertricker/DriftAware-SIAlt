"""Prepare frozen runs, execute them locally, or hand their snapshots to SLURM."""

from copy import deepcopy
from bisect import bisect_left
import datetime as dt
import math
from pathlib import Path
import subprocess
import uuid

import yaml

from driftaware_sialt.config_loader import load_config
from driftaware_sialt.io_tools import init_logger
from driftaware_sialt.input_data import required_input_dates
from driftaware_sialt.products.sea_ice_thickness import SeaIceThicknessMultiProducts, altimetry_source
from driftaware_sialt.products.sea_ice_concentration import SeaIceConcentrationProducts
from driftaware_sialt.products.sea_ice_drift import SeaIceDriftProducts
from driftaware_sialt.stacking.periods import DAY, RunPeriod, plan_periods


SNAPSHOT_VERSION = 1
SNAPSHOT_NAME = 'config_resolved.yaml'


def read_yaml(path):
    with Path(path).open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'expected a YAML mapping: {path}')
    return value


def _write_yaml(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x') as stream:
            yaml.safe_dump(value, stream, sort_keys=False)
    else:
        temporary = path.with_suffix('.yaml.tmp')
        with temporary.open('w') as stream:
            yaml.safe_dump(value, stream, sort_keys=False)
        temporary.replace(path)


def _git_commit():
    try:
        result = subprocess.run(
            ['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def read_snapshot(path, stage=None):
    path = Path(path).resolve()
    config = read_yaml(path)
    if config.get('snapshot_version') != SNAPSHOT_VERSION:
        raise ValueError(f'unsupported or missing snapshot_version: {path}')
    if config.get('stage') not in ('stacking', 'gridding'):
        raise ValueError(f'invalid snapshot stage: {path}')
    if stage is not None and config['stage'] != stage:
        raise ValueError(f'{stage} requires a {stage} snapshot: {path}')
    period = RunPeriod.from_config(config)
    if Path(config['run']['directory']).resolve() != path.parent:
        raise ValueError(f'snapshot directory does not match its location: {path}')
    if not config['run'].get('id') or not period.length:
        raise ValueError(f'incomplete run snapshot: {path}')
    return config


def _name_run(config):
    """Name a run without allocating its scientific processing grid."""
    run = config['run']
    stage = config['stage']
    options = config['options']
    settings = config[stage]
    grid = settings['parcel_grid' if stage == 'stacking' else 'netcdf_grid']
    if grid['dim'] < 1 or grid['bounds'][2] <= grid['bounds'][0]:
        raise ValueError('invalid processing grid')
    width = (grid['bounds'][2] - grid['bounds'][0]) / grid['dim'] * math.sqrt(2)
    if stage == 'stacking':
        window, mode = settings['t_window'], settings['mode']
        root = config['output_dir']['trajectories']
    else:
        source = config['run']['source_stack']
        window, mode = source['stacking']['t_window'], settings['mode']
        root = config['output_dir']['gridded_data']
    run['id'] = uuid.uuid4().hex[:12]
    period = RunPeriod.from_config(config)
    run['label'] = period.label
    offset = f"_{settings['dt_days_max']}" if stage == 'gridding' else ''
    name = (f"{options['target_variable']}-{options['hemisphere']}-{window}{mode}-"
            f"epsg{options['out_epsg'].split(':')[1]}_{width / 100:.0f}{offset}-"
            f"{period.label}-{dt.datetime.now():%Y-%m-%d_%H-%M-%S}-{run['id']}")
    run['directory'] = str(Path(root, name).resolve())
    config['logging'] = str(Path(run['directory'], 'processing.log'))
    config['snapshot_version'] = SNAPSHOT_VERSION
    config['software_git_commit'] = _git_commit()


def stacking_configs(config_file, *, hemisphere, sensors, years=None, start=None, end=None):
    config = load_config(config_file, hemisphere=hemisphere, sensors=sensors)
    if config['stage'] != 'stacking':
        raise ValueError('stacking/syncdata requires stage: stacking')
    if len(set(sensors)) != len(sensors):
        raise ValueError('duplicate sensors are not allowed')
    if 'icesat2' in sensors and len(sensors) != 1:
        raise ValueError('ICESat-2 must be processed separately from CCI missions')
    product = SeaIceThicknessMultiProducts(
        hem=hemisphere, sensor=sensors, target_var=config['options']['target_variable'])
    for sensor in sensors:
        if (sensor, product.target_var) not in product.function_map:
            raise ValueError(f'unsupported sensor/target combination: {sensor}/{product.target_var}')
    settings = config['stacking']
    periods = plan_periods(hemisphere, settings['t_window'], settings['mode'],
                           years=years, start=start, end=end)
    configs = []
    for period in periods:
        planned = deepcopy(config)
        planned['config_source'] = str(Path(config_file).resolve())
        planned['run'] = period.to_dict()
        _name_run(planned)
        configs.append(planned)
    return configs


def source_snapshots(*, source_run=None, source_batch=None):
    if bool(source_run) == bool(source_batch):
        raise ValueError('provide exactly one of --source-run or --source-batch')
    if source_batch:
        manifest = read_yaml(source_batch)
        if manifest.get('batch_version') != 1 or manifest.get('stage') != 'stacking':
            raise ValueError('gridding requires a stacking batch manifest')
        paths = [Path(entry['snapshot']) for entry in manifest['runs']]
        if not paths or len(set(paths)) != len(paths):
            raise ValueError('empty or duplicate runs in source batch')
    else:
        paths = [Path(source_run).resolve() / SNAPSHOT_NAME]
    snapshots = []
    for path in paths:
        source = read_snapshot(path, 'stacking')
        status_path = path.parent / 'status.yaml'
        status = read_yaml(status_path) if status_path.exists() else {}
        if status.get('state') != 'completed' or status.get('run_id') != source['run']['id']:
            raise ValueError(f'stacking run has not completed successfully: {path.parent}')
        if not _output_files(source):
            raise ValueError(f'completed stacking run has no trajectory files: {path.parent}')
        snapshots.append(source)
    return snapshots


def _output_files(config):
    directory = Path(config['run']['directory'])
    if config['stage'] == 'gridding':
        return list(directory.rglob('*.nc'))
    files = list(directory.glob('*.csv'))
    if config['stacking']['mode'] == 'fr':
        files = [path for path in files if '_F-' not in path.name and '_R-' not in path.name]
    return files


def gridding_configs(config_file, *, source_run=None, source_batch=None):
    configs = []
    for source in source_snapshots(source_run=source_run, source_batch=source_batch):
        options = source['options']
        config = load_config(config_file, hemisphere=options['hemisphere'], sensors=options['sensor'])
        if config['stage'] != 'gridding':
            raise ValueError('gridding requires stage: gridding')
        config['options']['target_variable'] = options['target_variable']
        config['options']['add_variable'] = options.get('add_variable', [])
        config['config_source'] = str(Path(config_file).resolve())
        period = RunPeriod.from_config(source)
        # Gridding consumes final target-date CSVs, not the stacking buffers.
        config['run'] = RunPeriod(period.output_start, period.output_end,
                                  period.output_start, period.output_end).to_dict()
        config['run']['source_run'] = source['run']['directory']
        # Include the small source settings needed for naming; the original
        # immutable snapshot remains the authoritative record of the input run.
        config['run']['source_stack'] = {'stacking': {
            't_window': source['stacking']['t_window'], 'mode': source['stacking']['mode']}}
        config['run']['source_snapshot'] = str(Path(source['run']['directory'], SNAPSHOT_NAME))
        config['run']['source_run_id'] = source['run']['id']
        _name_run(config)
        configs.append(config)
    return configs


def preview(configs, *, check_inputs=True):
    """Report planned dates and file coverage, without creating directories."""
    available = None
    auxiliary = {}
    if check_inputs and configs[0]['stage'] == 'stacking':
        config = configs[0]
        options = config['options']
        product = SeaIceThicknessMultiProducts(
            hem=options['hemisphere'], sensor=options['sensor'], target_var=options['target_variable'])
        product.get_file_list(config['input_dir'])
        product.get_file_dates()
        available = {date.date() for sensor in options['sensor'] for date in product.file_dates[sensor]}
        sources = dict.fromkeys(altimetry_source(s, config['input_dir']) for s in options['sensor'])
        print('Altimetry repositories:', {s: config['input_dir'][s] for s in sources})
        for family, cls, key in (
                ('ice_conc', SeaIceConcentrationProducts, 'ice_conc_products'),
                ('ice_drift', SeaIceDriftProducts, 'ice_drift_products')):
            directories = config['auxiliary'][family]
            products = cls.load_products(options[key], directories, hem=options['hemisphere'],
                                         out_epsg=options['out_epsg'])
            auxiliary[family] = sorted({date for product in products for date in product.file_dates})
            print(f'{family} repositories:', {s: directories[s] for s in options[key]})
    for config in configs:
        period = RunPeriod.from_config(config)
        options = config['options']
        print(f"{config['stage']} {options['hemisphere']} {'+'.join(options['sensor'])}: "
              f"outputs {period.output_start}..{period.output_end}; "
              f"processing {period.processing_start}..{period.processing_end}")
        print('  Output:', config['run']['directory'])
        if available is not None:
            output_dates = {period.output_start + index * DAY
                            for index in range((period.output_end - period.output_start).days + 1)}
            processing_dates = {period.processing_start + index * DAY for index in range(period.length)}
            missing = output_dates - available
            missing_buffer = (processing_dates - output_dates) - available
            if missing or missing_buffer:
                print(f'  Coverage: {len(missing)} output days and {len(missing_buffer)} '
                      'buffer days have no indexed altimetry files; stacks may be incomplete.')
            required = required_input_dates(config)
            for family, indexed in auxiliary.items():
                missing = 0
                for date in required[family]:
                    index = bisect_left(indexed, date)
                    nearest = indexed[max(0, index - 1):index + 1]
                    if not any(abs(candidate - date) <= 7 * DAY for candidate in nearest):
                        missing += 1
                if missing:
                    print(f'  Coverage: {family} unavailable for {missing} requested days '
                          '(including the seven-day fallback).')
            if config['stacking'].get('thermo_change', {}).get('enabled'):
                print('  Temperature/ocean heat flux contents are checked during processing.')


def prepare_batch(configs, execution):
    """Persist every snapshot before starting local work or submitting any job."""
    stage = configs[0]['stage']
    root = Path(configs[0]['output_dir']['trajectories' if stage == 'stacking' else 'gridded_data'])
    label = (f"{configs[0]['run']['output_start'][:4]}-{configs[-1]['run']['output_start'][:4]}"
             if len(configs) > 1 else configs[0]['run']['label'])
    batch_id = f'{stage}-{configs[0]["options"]["hemisphere"]}-{label}-{uuid.uuid4().hex[:12]}'
    manifest_path = root / 'batches' / f'{batch_id}.yaml'
    manifest = {'batch_version': 1, 'id': batch_id, 'stage': stage,
                'execution': execution, 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                'runs': []}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for config in configs:
        directory = Path(config['run']['directory'])
        directory.mkdir(parents=True, exist_ok=False)
        snapshot = directory / SNAPSHOT_NAME
        _write_yaml(snapshot, config, exclusive=True)
        _write_yaml(directory / 'status.yaml', {'state': 'prepared', 'run_id': config['run']['id']})
        manifest['runs'].append({'run_id': config['run']['id'], 'label': config['run']['label'],
                                 'directory': str(directory), 'snapshot': str(snapshot)})
    _write_yaml(manifest_path, manifest, exclusive=True)
    print('Batch manifest:', manifest_path, flush=True)
    return manifest_path, manifest


def execute_snapshot(path, run_stage):
    config = read_snapshot(path)
    directory = Path(config['run']['directory'])
    # Claim this output directory once, including after failures. Re-running
    # should create a fresh run, never mix a partial run with new results.
    with (directory / 'execution.lock').open('x') as stream:
        stream.write(config['run']['id'] + '\n')
    status = {'state': 'running', 'run_id': config['run']['id']}
    _write_yaml(directory / 'status.yaml', status)
    try:
        init_logger(config)
        if config['stage'] == 'gridding':
            source = source_snapshots(source_run=config['run']['source_run'])[0]
            if source['run']['id'] != config['run']['source_run_id']:
                raise ValueError('source stacking run changed after gridding was planned')
        run_stage(config)
        if not _output_files(config):
            raise RuntimeError(f'processing produced no final {config["stage"]} outputs: {directory}')
    except BaseException as error:
        status.update(state='failed', error=str(error))
        _write_yaml(directory / 'status.yaml', status)
        raise
    status['state'] = 'completed'
    _write_yaml(directory / 'status.yaml', status)


def submit_batch(manifest_path, manifest, script):
    """Submit explicit snapshot arguments; never invoke a shell for sbatch."""
    script = str(Path(script).resolve())
    for entry in manifest['runs']:
        directory = Path(entry['directory'])
        command = [
            'sbatch', '--parsable',
            f"--job-name=cci_{manifest['stage']}_{entry['label']}_{manifest['id'].split('-')[1]}",
            f'--output={directory}/slurm-%x-%j.out',
            f'--error={directory}/slurm-%x-%j.err', script, entry['snapshot']]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(';')[0]
            if not job_id.isdigit():
                raise RuntimeError(f'unexpected sbatch response: {result.stdout!r}')
            entry['job_id'] = job_id
            print(f"Submitted {entry['label']}: job {job_id}", flush=True)
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            entry['submission_error'] = str(error)
            _write_yaml(manifest_path, manifest)
            raise
        _write_yaml(manifest_path, manifest)
