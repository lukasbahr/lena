import yaml
import logging
import sys
import pprint
from pathlib import Path

from lena.util.configlib import config as args
import lena.util.configlib as configlib
from lena.net.train import train
from lena.net.helperfnc import generateTrainingData, processModel
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
    if args['debug']:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info("Running eperiment in path: {}".format(args['exp_config']))

	# Open experiment config
    with open(args['exp_config'], 'r') as file:

        # Load params for specified experiment 
        params = {}
        load = yaml.load(file, Loader=yaml.FullLoader)
        for d in load:
            params.update(d)
        path = Path(args['exp_config']).parent.absolute()
        params.update({'path':str(path)})
        pprint.pprint(params)

        observer = createDefaultObserver(params['system'])
        data = generateTrainingData(observer, params['data'])
        model = train(data, observer, params['model'])

        if params['write_experiment']:
            processModel(data,observer, model, params)



