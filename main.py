import logging
import sys
import pprint

from lena.util.configlib import config as args
import lena.util.configlib as configlib
from lena.util.params import Params
from lena.net.train import train
from lena.net.helperfnc import generateTrainingData, processModel, generateMesh
from lena.datasets.exampleSystems import createDefaultObserver
from sklearn.manifold import TSNE

def plotTSNE(observer,model,params):
    import matplotlib.pyplot as plt

    data_val=generateTrainingData(observer, params['validation'])

    z, x_hat=model(data_val[:, :observer.dim_x+observer.optionalDim])

    tsne = TSNE(random_state=123).fit_transform(z.detach().numpy())
    plt.scatter(tsne[:,0], tsne[:,1])
    plt.show()

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

        paramsHandler = Params(file, args)
        params = paramsHandler.params

        observer = createDefaultObserver(params['system'])
        data = generateTrainingData(observer, params['data'])
        model = train(data, observer, params['model'])

        plotTSNE(observer, model, params)

        if params['write_experiment']:
            processModel(data,observer, model, params)


