import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import time


def rms_error(x, x_hat):
    return np.log(
        np.sqrt(
            (np.power((x[:, 0]-x_hat[:, 0]), 2) + np.power((x[:, 1]-x_hat[:, 1]), 2)) /
            x[0].shape[0]
        )
    )


def plotLogError2D(x, x_hat, mesh, params):
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Create matplot figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    pos = ax.scatter(mesh[:, 0], mesh[:, 1], cmap='jet',
                     c=rms_error(x, x_hat))

    fig.colorbar(pos, ax=ax)
    fig.set_label('Log relative error')

    if params['write_experiment']:
        fig.savefig(params['path']+'/'+timestr+'_heatmap.png', dpi=300)
    else:
        plt.show()


def plotTrajectory2D(x):
    plt.scatter(x[0], x[1])
    plt.show()


def plotSimulation2D(tq, x, x_hat, params, idx=0):
    timestr = time.strftime("%Y%m%d-%H%M%S")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title('Simulation for true and estimated initial conditions')
    ax.set_ylabel('state')
    ax.set_xlabel('time')

    ax.plot(tq, x, color='blue', label='x')
    ax.plot(tq, x_hat, color='red', linestyle='dashed',label='x_hat')


    if params['write_experiment']:
        fig.savefig(params['path']+'/'+timestr+'_simulation_'+str(idx)+'.png', dpi=300)
    else:
        plt.show()
