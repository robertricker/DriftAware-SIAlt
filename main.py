import datetime
import time
import yaml
import argparse
import os
from typing import Dict
from loguru import logger
from config_loader import load_config


def main(configure: Dict[str, object]) -> None:
    logger.info('configuration settings:\n{}'.format(yaml.dump(configure)))

    stage = configure["stage"]

    if stage == 'stacking':
        from stacking import stacking

        logger.info('start stacking')
        stacking.stacking(configure)
        logger.info('finished stacking')

    elif stage == 'gridding':
        from gridding import gridding

        logger.info('start evaluation on grid')
        gridding.gridding(configure)
        logger.info('finished evaluation on grid')

    elif stage == 'visualization':
        from visualization import visualization

        logger.info('start visualization')
        visualization.visualization(configure)
        logger.info('finished visualization')

    elif stage == 'volume':
        from volume import volume

        logger.info('start volume computation')
        volume.volume(configure)
        logger.info('finished volume computation')

    else:
        raise ValueError('unexpected processing stage: %s' % stage)

    elapsed_time = time.time() - start_time
    time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    logger.info("elapsed_time: %s" % time_str)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process configuration file.')
    parser.add_argument('config_file', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()

    start_time = time.time()
    config = load_config(args.config_file)

    # Set up the logging configuration
    log_file = f"{config['stage']}{'_'}"\
               f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    config['logging'] = os.path.join(config['logging'], log_file)
    from io_tools import init_logger

    init_logger(config)
    main(config)
