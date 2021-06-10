import yaml
import logging
import sys
import pprint

from lena.tools.configlib import config as params
import lena.tools.configlib as configlib
from lena.tools.map import Map
from lena.net.train import trainAutonomousAutoencoder
from lena.net.helperfnc import generateTrainingData
from lena.datasets.exampleSystems import createDefaultObserver


# Configuration arguments
parser = configlib.add_parser("Train config")
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
parser.add_argument("--exp_config", type=str, help="Config file for the experiment.")

# Configure logger
logging.basicConfig(format='%(asctime)s %(message)s')

if __name__ == "__main__":
    # Save terminal args
    configlib.parse(save_fname="last_arguments.txt")

    # Get logger
    logger = logging.getLogger(__name__)

# Set logging level
    if params['debug']:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info("Running eperiment in path: {}".format(params['exp_config']))

	# Open experiment config
    with open(params['exp_config'], 'r') as file:

        # Load params for specified experiment 
        params = {}
        load = yaml.load(file, Loader=yaml.FullLoader)
        for d in load:
            params.update(d)
        pprint.pprint(params)

        observer = createDefaultObserver(params['system'])
        data = generateTrainingData(observer, params['data'])
        trainAutonomousAutoencoder(data, observer, params)
