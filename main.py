import yaml
import logging
import sys

from lena.tools.configlib import config as params
import lena.tools.configlib as configlib
from lena.tools.map import Map

# Configuration arguments
parser = configlib.add_parser("Train config")
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
parser.add_argument("--exp_config", type=str, help="Config file for the experiment.")

if __name__ == "__main__":
	configlib.parse(save_fname="last_arguments.txt")

	logging.basicConfig(format='%(asctime)s %(message)s')
	logger = logging.getLogger(__name__)

	if params['debug']:
		logger.setLevel(logging.DEBUG)
	else:
		logger.setLevel(logging.INFO)

	logger.info("Running with configuration:")
	configlib.print_config()

	with open(params['exp_config'], 'r') as file:
		params = yaml.load(file, Loader=yaml.FullLoader)
		params = Map(params[0])
		logger.info(params.model)
