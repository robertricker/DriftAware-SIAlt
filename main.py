import datetime
import time
import yaml
import argparse
from gridding import gridding
from visualization import visualization
from typing import Dict
from loguru import logger
from stacking import stacking
from io_tools import init_logger


def main(configure: Dict[str, object]) -> None:
    logger.info('configuration settings:\n{}'.format(yaml.dump(configure)))

    proc_step = configure["options"]["proc_step"]

    if proc_step == 'stacking':
        logger.info('start stacking')
        stacking.stacking(configure)
        logger.info('finished stacking')

    elif proc_step == 'gridding':
        logger.info('start evaluation on grid')
        gridding.gridding(configure)
        logger.info('finished evaluation on grid')

    elif proc_step == 'visualization':
        logger.info('start visualization')
        visualization.visualization(configure)
        logger.info('finished visualization')

    else:
        raise ValueError('unexpected proc_step: %s' % proc_step)

    elapsed_time = time.time() - start_time
    time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    logger.info("elapsed_time: %s" % time_str)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process configuration file.')
    parser.add_argument('config_file', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()

    start_time = time.time()
    # Load the configuration settings from a YAML file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Set up the logging configuration
    log_file = f"{config['dir']['logging']}{config['user']['name']}{'_'}" \
               f"{config['options']['proc_step']}{'_'}"\
               f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    config['dir']['logging'] = log_file
    init_logger(config)
    main(config)
