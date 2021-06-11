import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


def rms_error(x, x_hat):
    return np.log(
        np.sqrt(
            (np.power((x[:,0]-x_hat[:,0]), 2) + np.power((x[:,1]-x_hat[:,1]), 2)) /
            x[0].shape[0]
        )
    )


def plotLogError2D(x, x_hat, mesh, params):

    # Create matplot figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    pos = ax.scatter(mesh[:, 0], mesh[:, 1], cmap='jet',
               c=rms_error(x, x_hat))

    fig.colorbar(pos, ax=ax)
    fig.set_label('Log relative error')

    if params['write_experiment']:
        fig.savefig(params['path']+'/heatmap.png', dpi=300)
    else:
        plt.show()


def plotTrajectory2D(x):
    plt.scatter(x[0], x[1])
    plt.show()


def plotSimulation2D(tq, x):
    plt.plot(tq, x)
    plt.show()
