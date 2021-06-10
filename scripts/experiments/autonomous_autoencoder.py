import sys
sys.path.append(sys.path[0]+'/../..')
import numpy as np
import torch
from lena.net.train import trainAutonomousAutoencoder
from lena.net.helperfnc import generateTrainingData
from lena.datasets.exampleSystems import createDefaultObserver
from lena.tools.configlib import config as params


def getOptions():
    """
    Configure model options for the experiment
    """
    options = {}

    options['batchSize'] = 3
    options['epochs'] = 100
    options['numHiddenLayers'] = 5
    options['sizeHiddenLayer'] = 30
    options['activation'] = 'tanh'
    options['reconLambda'] = .1
    options['isTensorboard'] = False
    options['shuffle'] = False

    options['simulationTime'] = 20
    options['simulationStep'] = 1e-2

    options['system'] = 'autonomous'
    options['isAutonomous'] = True

    # options['dataGen'] = 'pairs'
    options['dataGen'] = 'pairs'
    options['sampling'] = 'lhs'
    options['gridSize'] = np.arange(-1, 1, 0.5)
    options['lhs_limits'] = np.array([[-1., 1.], [-1., 1.]])
    options['lhs_samples'] = 2

    return options


if __name__ == "__main__":

    options = getOptions()
    observer = createDefaultObserver(options)
    data = generateTrainingData(observer, options)
    trainAutonomousAutoencoder(data, observer, options)
    
