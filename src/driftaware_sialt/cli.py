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


def run(configure: Dict[str, object]) -> None:
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
        description="Run DriftAware-SIAlt processing."
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the processing configuration YAML file.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse command-line arguments and run the selected stage."""
    args = build_parser().parse_args(argv)
    start_time = time.time()
    config = load_config(args.config_file)

    log_file = (
        f"{config['stage']}_"
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    )
    config["logging"] = os.path.join(config["logging"], log_file)
    init_logger(config)

    run(config)

    elapsed_time = time.time() - start_time
    time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    logger.info("elapsed_time: %s" % time_str)
    return 0
