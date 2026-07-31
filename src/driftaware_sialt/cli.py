"""Command-line entry point for DriftAware-SIAlt."""

import argparse
import datetime
import os
import time
from typing import Dict, Optional, Sequence

import yaml
from loguru import logger

from driftaware_sialt.config_loader import load_config
from driftaware_sialt.io_tools import init_logger


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize data or run a DriftAware-SIAlt stage."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sync_parser = commands.add_parser(
        "syncdata",
        help="Synchronize inputs required by a stacking configuration.",
    )
    sync_parser.add_argument(
        "config_file",
        type=str,
        help="Path to the stacking configuration YAML file.",
    )

    for stage in PROCESSING_STAGES:
        stage_parser = commands.add_parser(
            stage,
            help=f"Run the {stage} processing stage.",
        )
        stage_parser.add_argument(
            "config_file",
            type=str,
            help=f"Path to the {stage} configuration YAML file.",
        )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse a synchronization or processing command and execute it."""
    args = build_parser().parse_args(argv)
    start_time = time.time()
    config = load_config(args.config_file)
    command = args.command

    if command == "syncdata" and config["stage"] != "stacking":
        raise ValueError(
            "syncdata requires a configuration with stage: stacking")
    if command in PROCESSING_STAGES and config["stage"] != command:
        raise ValueError(
            f"{command} command requires a configuration with "
            f"stage: {command}; got stage: {config['stage']}")

    log_file = (
        f"{command}_{config['stage']}_"
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    )
    config["logging"] = os.path.join(config["logging"], log_file)
    init_logger(config)

    if command == "syncdata":
        from driftaware_sialt.input_data import sync_required_input_data

        logger.info("configuration settings:\n{}".format(yaml.dump(config)))
        logger.info("start input repository synchronization")
        report = sync_required_input_data(config)
        logger.info(
            "finished input repository synchronization: {} files downloaded",
            len(report.downloaded),
        )
    else:
        run_stage(config)

    elapsed_time = time.time() - start_time
    time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    logger.info("elapsed_time: %s" % time_str)
    return 0
