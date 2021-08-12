from lena.experiments.noise import run_noise_exp
from lena.experiments.autonomous import run_autonomous_exp
from lena.experiments.non_autonomous import run_non_autonomous_exp
from lena.util.configlib import config as args
import lena.util.configlib as configlib
from lena.util.params import Params


# Configuration arguments
parser = configlib.add_parser("Train config")
parser.add_argument("--exp_config", type=str, help="Config file for the experiment.")

if __name__ == "__main__":
    # Save terminal args
    configlib.parse(save_fname="last_arguments.txt")

    # Open experiment config
    with open(args['exp_config'], 'r') as file:

        paramsHandler = Params(file, args)
        params = paramsHandler.params

        if params['data']['experiment'] == 'noise':
            run_noise_exp(params)
        elif params['data']['experiment'] == 'autonomous':
            run_autonomous_exp(params)
        elif params['data']['experiment'] == 'non_autonomous':
            run_non_autonomous_exp(params)