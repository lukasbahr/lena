import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import time
from sklearn.manifold import TSNE


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
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    pos = ax.scatter(mesh[:, 0], mesh[:, 1], cmap='jet',
                     c=rms_error(x, x_hat))

    fig.colorbar(pos, ax=ax)
    fig.set_label('Log relative error')

    if params['write_experiment']:
        fig.savefig(params['path']+'/'+timestr+'_heatmap.png', dpi=300)
    else:
        plt.show()

    return fig, ax


def plotTrajectory2D(x):
    plt.scatter(x[0], x[1])
    plt.show()


def plotSimulation2D(tq, x, writeExperiment=False, path="", x_hat=0, idx=0):
    timestr = time.strftime("%Y%m%d-%H%M%S")

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title('Simulation for true and estimated initial conditions')
    ax.set_ylabel('state')
    ax.set_xlabel('time' + r'$[s]$')

    ax.plot(tq, x, color='blue', label='x')

    if x_hat != 0:
        ax.plot(tq, x_hat, color='red', linestyle='dashed', label='x_hat')

    if writeExperiment:
        fig.savefig(path+'/'+timestr+'_simulation_'+str(idx)+'.png', dpi=300)
    else:
        plt.show()

    return fig, ax


def plotTSNE(z, writeExperiment=False, path=""):

    timestr = time.strftime("%Y%m%d-%H%M%S")

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title('Simulation for true and estimated initial conditions')
    ax.set_ylabel('state')
    ax.set_xlabel('time' + r'$[s]$')

    tsne = TSNE(random_state=123).fit_transform(z)
    ax.scatter(tsne[:, 0], tsne[:, 1])

    if writeExperiment:
        fig.savefig(path+'/'+timestr+'_tsne.png', dpi=300)
    else:
        plt.show()

    return fig, ax


def plotObserverEstimation2D(tq, x, x_hat, writeExperiment=False, path=""):
    timestr = time.strftime("%Y%m%d-%H%M%S")

    condstr = str(x[0, 0]) + "_" + str(x[0, 1])

    # Create fig with 300 dpi
    fig = plt.figure(dpi=200)

    # Create ax_trans figure
    ax_x1 = fig.add_subplot(3, 1, 1)
    ax_x1.plot(tq, x_hat[:, 0].to("cpu"), color='red', linestyle='dashed', label='x_hat')
    ax_x1.plot(tq, x[:, 0], color='blue', label='x')

    ax_x1.set_ylabel(r'$x_1$')
    ax_x1.set_xlabel('time' + r'$[s]$')

    # Create ax_trans figure
    ax_x2 = fig.add_subplot(3, 1, 2)
    ax_x2.plot(tq, x_hat[:, 1].to("cpu"), color='red', linestyle='dashed', label='x_hat')
    ax_x2.plot(tq, x[:, 1], color='blue', label='x')

    ax_x2.set_ylabel(r'$x_2$')
    ax_x2.set_xlabel('time' + r'$[s]$')

    # Create ax_trans figure
    ax_error = fig.add_subplot(3, 1, 3)
    ax_error.plot(tq, sqrtError(x_hat[:, 0], x[:, 0]), color='red', label='x_hat')
    ax_error.plot(tq, sqrtError(x_hat[:, 1], x[:, 1]), color='blue', label='x')

    ax_error.set_ylabel(r'$x_i-\hat{x_i}$')
    ax_error.set_xlabel('time' + r'$[s]$')

    fig.tight_layout()

    if writeExperiment:
        fig.savefig(path+'/'+timestr+'_'+condstr + '_simulation.png', dpi=300)
    else:
        plt.show()

    return fig


def sqrtError(x, x_hat):
    return x-x_hat
