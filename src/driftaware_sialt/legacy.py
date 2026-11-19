"""Register finished CSV archives without inventing their historical settings."""

import csv
import datetime as dt
import hashlib
from pathlib import Path
import re
import sys
import uuid

from driftaware_sialt.io_tools import read_dasit_metadata
from driftaware_sialt import workflow


INSTRUMENTS = {
    'RA_ERS2': 'ers2', 'RA2_ENVISAT': 'envisat', 'SIRAL_CRYOSAT2': 'cryosat2',
    'SRAL_SENTINEL3A': 'sentinel3a', 'SRAL_SENTINEL3B': 'sentinel3b', 'ATLAS_ICESAT2': 'icesat2',
}
TARGETS = {'SITHICK': 'sea_ice_thickness', 'SIFB': 'sea_ice_freeboard',
           'RFB': 'radar_freeboard', 'TFB': 'total_freeboard'}
FILENAME = re.compile(
    r'^.+-L2P-DA(?:_(?P<direction>F|R))?-(?P<target>SITHICK|SIFB|RFB|TFB)-'
    r'(?P<instruments>.+)-(?P<hemisphere>NH|SH)-(?P<date>\d{8})-fv(?P<version>[^/]+)\.csv$')


def inventory(directory):
    """Detect changed/added/removed CSVs cheaply, without hashing large data files."""
    digest = hashlib.sha256()
    for path in sorted(Path(directory).glob('*.csv')):
        stat = path.stat()
        digest.update(f'{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n'.encode())
    return digest.hexdigest()


def verify_import(config):
    if inventory(config['run']['directory']) != config['legacy_import']['csv_inventory_sha256']:
        raise ValueError(f"legacy CSV files changed after registration: {config['run']['directory']}")


def inspect_legacy(directory):
    # Trajectory geometry cells can exceed the csv module's default 128 KiB limit.
    csv.field_size_limit(sys.maxsize)
    directory = Path(directory).resolve()
    signature = inventory(directory)
    files = sorted(directory.glob('*.csv'))
    dates, identity, columns = [], None, None
    for path in files:
        match = FILENAME.fullmatch(path.name)
        if not match:
            raise ValueError(f'unsupported trajectory filename: {path}')
        metadata = read_dasit_metadata(path)
        stack = metadata.get('source_stack')
        if not stack:
            raise ValueError(f'{path}: registration requires the JSON source_stack CSV header')
        mode = stack['mode']
        if mode not in ('f', 'r', 'fr') or match['direction'] != {'f': 'F', 'r': 'R', 'fr': None}[mode]:
            raise ValueError(f'{path}: incomplete directional output or inconsistent stack mode')
        window = stack['window_days']
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError(f'{path}: invalid stacking window')
        histogram = stack['histogram']
        if not histogram.get('n_bins') or not histogram.get('range'):
            raise ValueError(f'{path}: missing histogram settings')
        hemisphere = match['hemisphere'].lower()
        crs = 'EPSG:6931' if hemisphere == 'nh' else 'EPSG:6932'
        target = TARGETS[match['target']]
        if metadata.get('crs') != crs or metadata.get('target_variable') != target:
            raise ValueError(f'{path}: filename and CSV metadata disagree')
        sensors = [sensor for tag, sensor in INSTRUMENTS.items() if tag in match['instruments']]
        remaining = match['instruments']
        for tag in INSTRUMENTS:
            remaining = remaining.replace(tag, '')
        if not sensors or remaining.strip('_'):
            raise ValueError(f'{path}: unknown mission identifiers')
        with path.open(newline='') as stream:
            stream.readline()
            reader = csv.reader(stream, strict=True)
            header, row = next(reader, None), next(reader, None)
        if not header or not row or len(header) != len(row):
            raise ValueError(f'{path}: empty or malformed CSV')
        if not {'geometry', 'dt_days', target, target + '_hist'} <= set(header):
            raise ValueError(f'{path}: missing required trajectory columns')
        current = (hemisphere, sensors, target, match['version'], stack)
        if identity is not None and (identity != current or columns != header):
            raise ValueError(f'{directory}: mixed mission, version, stack metadata, or column schemas')
        identity, columns = current, header
        dates.append(dt.datetime.strptime(match['date'], '%Y%m%d').date())
    if not files or len(set(dates)) != len(dates):
        raise ValueError(f'{directory}: no final CSVs or duplicate target dates')
    if inventory(directory) != signature:
        raise ValueError(f'{directory}: CSVs changed during inspection; wait until processing finishes')
    hemisphere, sensors, target, version, stack = identity
    begin, end = min(dates).isoformat(), max(dates).isoformat()
    additional = [name for name in ('snow_depth', 'radar_freeboard', 'radar_freeboard_l2_unc',
                                   'sea_ice_freeboard', 'sea_ice_freeboard_l2_unc')
                  if name in columns and name not in (target, target + '_l2_unc')]
    return {
        'snapshot_version': workflow.SNAPSHOT_VERSION, 'stage': 'stacking', 'version': version,
        'software_git_commit': None,
        'options': {'hemisphere': hemisphere, 'out_epsg': crs, 'sensor': sensors,
                    'target_variable': target, 'add_variable': additional},
        'stacking': {'mode': stack['mode'], 't_window': stack['window_days'], 'hist': stack['histogram']},
        'run': {'id': 'legacy-' + uuid.uuid4().hex[:12], 'label': f'{begin}_{end}',
                'directory': str(directory), 'output_start': begin, 'output_end': end,
                'processing_start': None, 'processing_end': None},
        'legacy_import': {
            'registered_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'purpose': 'Gridding-only registration reconstructed from existing final CSVs.',
            'csv_count': len(files), 'csv_inventory_sha256': signature,
            'date_basis': 'Observed CSV target dates; original requested/padded dates are unknown.',
            'unknown_original_settings': ['software_git_commit', 'processing dates',
                                          'repositories', 'filtering and growth configuration'],
        },
    }


def register_trajectories(source, *, completed=False, dry_run=False):
    if not completed and not dry_run:
        raise ValueError('use --completed only after confirming these legacy runs have finished')
    root = Path(source).resolve()
    if not root.is_dir():
        raise ValueError(f'trajectory directory does not exist: {root}')
    single = any(root.glob('*.csv'))
    directories = [root] if single else sorted(p for p in root.iterdir() if p.is_dir() and p.name != 'batches')
    records = []
    for directory in directories:
        markers = [directory / workflow.SNAPSHOT_NAME, directory / 'status.yaml', directory / 'execution.lock']
        if any(path.exists() for path in markers):
            if not all(path.exists() for path in markers[:2]):
                raise ValueError(f'partial registration already exists: {directory}')
            record, = workflow.source_snapshots(source_run=directory)
            print('Already registered:', directory)
        elif any(directory.glob('*.csv')):
            record = inspect_legacy(directory)
            print(f"Register: {directory.name}: {record['legacy_import']['csv_count']} CSVs, "
                  f"{record['run']['output_start']}..{record['run']['output_end']}")
        else:
            print('Skipping empty directory:', directory)
            continue
        records.append(record)
    if not records:
        raise ValueError('no finished trajectory runs found')
    if dry_run:
        return []
    groups = {}
    for record in records:
        directory = Path(record['run']['directory'])
        if not (directory / workflow.SNAPSHOT_NAME).exists():
            verify_import(record)
            workflow._write_yaml(directory / workflow.SNAPSHOT_NAME, record, exclusive=True)
            workflow._write_yaml(directory / 'status.yaml', {
                'state': 'completed', 'run_id': record['run']['id'],
                'completion_basis': 'User-confirmed finished legacy CSV archive; not a recovered SLURM status.',
            }, exclusive=True)
        key = (record['options']['hemisphere'], tuple(record['options']['sensor']))
        groups.setdefault(key, []).append(record)
    manifest_paths = []
    for (hemisphere, _), group in groups.items():
        batch_id = f'stacking-{hemisphere}-legacy-{uuid.uuid4().hex[:12]}'
        path = (root.parent if single else root) / 'batches' / f'{batch_id}.yaml'
        path.parent.mkdir(exist_ok=True)
        manifest = {'batch_version': 1, 'id': batch_id, 'stage': 'stacking',
                    'execution': 'legacy-import', 'runs': [
                        {'run_id': c['run']['id'], 'label': c['run']['label'],
                         'directory': c['run']['directory'],
                         'snapshot': str(Path(c['run']['directory'], workflow.SNAPSHOT_NAME))}
                        for c in sorted(group, key=lambda c: c['run']['output_start'])]}
        workflow._write_yaml(path, manifest, exclusive=True)
        manifest_paths.append(path)
        print('Batch manifest:', path)
    return manifest_paths
