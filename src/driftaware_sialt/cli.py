"""Command-line entry point for DriftAware-SIAlt."""

import argparse
import datetime
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import yaml
from loguru import logger

from driftaware_sialt.config_loader import load_config
from driftaware_sialt.io_tools import init_logger
from driftaware_sialt import workflow


PROCESSING_STAGES = ("stacking", "gridding", "visualization", "volume")


def run_stage(configure: Dict[str, object]) -> None:
    """Run the processing stage selected by a resolved configuration."""
    logger.info("configuration settings:\n{}".format(yaml.dump(configure)))

    stage = configure["stage"]

    if stage == "stacking":
        from driftaware_sialt.stacking import pipeline

        logger.info("start stacking")
        pipeline.run(configure)
        logger.info("finished stacking")

    elif stage == "gridding":
        from driftaware_sialt.gridding import gridding

        logger.info("start evaluation on grid")
        gridding.gridding(configure)
        logger.info("finished evaluation on grid")

    elif stage == "visualization":
        from driftaware_sialt.visualization import visualization

        logger.info("start visualization")
        visualization.visualization(configure)
        logger.info("finished visualization")

    elif stage == "volume":
        from driftaware_sialt.volume import pipeline

        logger.info("start volume computation")
        pipeline.run(configure)
        logger.info("finished volume computation")

    else:
        raise ValueError("unexpected processing stage: %s" % stage)


def _add_run_arguments(parser, stage):
    parser.add_argument('config_file', help='Scientific configuration YAML or a resolved snapshot.')
    parser.add_argument('--dry-run', action='store_true', help='Preview without writing files or submitting jobs.')
    if stage == 'gridding':
        sources = parser.add_mutually_exclusive_group()
        sources.add_argument('--source-run', help='Completed stacking run directory.')
        sources.add_argument('--source-batch', help='Manifest from a completed stacking batch.')
    else:
        parser.add_argument('--hemisphere', choices=('nh', 'sh'))
        parser.add_argument('--sensors', nargs='+', help='Mission names, or multi-mission for all supported CCI platforms.')
        if stage in ('stacking', 'syncdata'):
            dates = parser.add_mutually_exclusive_group()
            dates.add_argument('--years', help='YYYY or inclusive YYYY:YYYY; NH uses winter starting years.')
            dates.add_argument('--start', help='First output date, YYYY-MM-DD (no automatic padding).')
            parser.add_argument('--end', help='Last output date, inclusive; required with --start.')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Plan, run, or submit DriftAware-SIAlt processing.')
    commands = parser.add_subparsers(dest='command', required=True)
    register = commands.add_parser('register-trajectories', help='Register finished pre-snapshot CSV runs for gridding.')
    register.add_argument('source', help='One legacy run directory or a parent containing run directories.')
    register.add_argument('--completed', action='store_true', help='Confirm that the selected legacy runs have finished.')
    register.add_argument('--dry-run', action='store_true', help='Inspect and validate without writing metadata.')
    for stage in (*PROCESSING_STAGES, 'syncdata'):
        _add_run_arguments(commands.add_parser(stage), stage)
    submit = commands.add_parser('submit', help='Freeze configurations and submit independent SLURM jobs.')
    stages = submit.add_subparsers(dest='stage', required=True)
    for stage in ('stacking', 'gridding'):
        stage_parser = stages.add_parser(stage)
        _add_run_arguments(stage_parser, stage)
        stage_parser.add_argument('--slurm-script', required=True, help='Batch script accepting a snapshot as $1.')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse a synchronization or processing command and execute it."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'register-trajectories':
        from driftaware_sialt.legacy import register_trajectories
        try:
            register_trajectories(args.source, completed=args.completed, dry_run=args.dry_run)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        return 0
    start_time = time.time()
    command = args.command
    stage = args.stage if command == 'submit' else command
    try:
        raw = workflow.read_yaml(args.config_file)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if 'snapshot_version' in raw:
        if command not in ('stacking', 'gridding'):
            parser.error('execute a snapshot with its stacking or gridding command')
        if any(getattr(args, key, None) for key in (
                'hemisphere', 'sensors', 'years', 'start', 'end', 'source_run', 'source_batch')):
            parser.error('a frozen snapshot cannot be overridden')
        config = workflow.read_snapshot(args.config_file, stage)
        if config.get('legacy_import'):
            parser.error('legacy registrations are gridding sources; use --source-run or --source-batch')
        if args.dry_run:
            workflow.preview([config])
        else:
            workflow.execute_snapshot(args.config_file, run_stage)
        return 0

    try:
        if stage == 'gridding':
            configs = workflow.gridding_configs(
                args.config_file, source_run=args.source_run, source_batch=args.source_batch)
        else:
            if not args.hemisphere or not args.sensors:
                raise ValueError('--hemisphere and --sensors are required')
            if stage in ('stacking', 'syncdata'):
                configs = workflow.stacking_configs(
                    args.config_file, hemisphere=args.hemisphere, sensors=args.sensors,
                    years=args.years, start=args.start, end=args.end)
            else:
                config = load_config(args.config_file, hemisphere=args.hemisphere, sensors=args.sensors)
                if config['stage'] != stage:
                    raise ValueError(f'{stage} requires stage: {stage}')
                configs = [config]
        if command == 'submit':
            if not Path(args.slurm_script).is_file():
                raise ValueError(f'SLURM script does not exist: {args.slurm_script}')
            if not args.dry_run and shutil.which('sbatch') is None:
                raise ValueError('sbatch was not found; use the local stacking/gridding command on this machine')
    except (ValueError, KeyError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))

    if stage in ('stacking', 'gridding', 'syncdata'):
        workflow.preview(configs)
    elif args.dry_run:
        print(yaml.safe_dump(configs[0], sort_keys=False))
    if args.dry_run:
        return 0

    if stage in ('stacking', 'gridding'):
        manifest_path, manifest = workflow.prepare_batch(configs, 'slurm' if command == 'submit' else 'local')
        if command == 'submit':
            workflow.submit_batch(manifest_path, manifest, args.slurm_script)
        else:
            for entry in manifest['runs']:
                workflow.execute_snapshot(entry['snapshot'], run_stage)
    else:
        log_root = load_config(args.config_file, hemisphere=args.hemisphere,
                               sensors=args.sensors)['logging']
        Path(log_root).mkdir(parents=True, exist_ok=True)
        for config in configs:
            # Synchronization has no processing outputs; use the configured log root.
            config['logging'] = os.path.join(log_root, f'{stage}_{time.time_ns()}.log')
            init_logger(config)
            if stage == 'syncdata':
                from driftaware_sialt.input_data import sync_required_input_data
                sync_required_input_data(config)
            else:
                run_stage(config)

    elapsed_time = time.time() - start_time
    time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    logger.info("elapsed_time: %s" % time_str)
    return 0
