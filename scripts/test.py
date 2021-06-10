import sys
sys.path.append(sys.path[0]+'/..')
from lena.tools.configlib import config as params
import lena.tools.configlib as configlib

# Configuration arguments
parser = configlib.add_parser("Train config")
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

if __name__ == "__main__":
    configlib.parse(save_fname="last_arguments.txt")
    print("Running with configuration:")
    configlib.print_config()
