import numpy as np
import torch
import random
import math
import dill as pickle
from scipy import linalg, signal
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
import time

import lena.util.plot as plot


def normalize(self, data):
    """
    Normalize data between [0,1]
    """
    return (data-np.min(data))/(np.max(data) - np.min(data))


def generateMesh(params):
    # Sample either a uniformly grid or use latin hypercube sampling
    if params['sampling'] == 'uniform':
        axes = np.arange(params['grid_size'][0], params['grid_size'][1], params['grid_step'])
        mesh = np.array(np.meshgrid(axes, axes)).T.reshape(-1, 2)
        mesh = torch.tensor(mesh)
    elif params['sampling'] == 'lhs':
        limits = np.array([params['lhs_limits'], params['lhs_limits']])
        sampling = LHS(xlimits=limits)
        mesh = torch.tensor(sampling(params['lhs_samples']))

    return mesh


def generateTrainingData(observer, params):
    """
    Generate training samples (x,z) by simulating backward in time
    and after forward in time.
    """

    mesh = generateMesh(params)
    nsims = mesh.shape[0]

    # Set simulation step width
    dt = params['simulation_step']

    # Generate either pairs of (x_i, z_i) values by simulating back and then forward in time
    # or generate trajectories for every initial value (x_1_i, x_2_i, z_0)
    if params['type'] == 'pairs':

        # Create dataframes
        y_0 = torch.zeros((observer.dim_x + observer.dim_z, 1))
        y_1 = torch.zeros((observer.dim_x + observer.dim_z, 1))
        data = torch.zeros((nsims, observer.dim_x + observer.dim_z + 1))

        for i in range(nsims):
            # Eigenvalues for D
            w_c = random.uniform(2.5, 2.5) * math.pi
            b, a = signal.bessel(3, w_c, 'low', analog=True, norm='phase')

            eigen = np.roots(a)

            # Place eigenvalue
            observer.D = observer.tensorDFromEigen(eigen)

            # Advance k/min(lambda) in time
            k = 40
            t_c = k/min(abs(observer.eigenD.real))
            t_c = 10

            # Simulate backward in time
            tsim = (0, -t_c)
            y_0[:observer.dim_x, :] = mesh[i].unsqueeze(1)
            tq_bw, data_bw = observer.simulateLueneberger(y_0, tsim, -dt)

            # Simulate forward in time starting from the last point from previous simulation
            tsim = (-t_c, 0)
            y_1[:observer.dim_x, :] = data_bw[-1, :observer.dim_x, :]
            tq, data_fw = observer.simulateLueneberger(y_1, tsim, dt)

            # Data contains (x_i, z_i, w_c_i) pairs in shape [1+dim_x+dim_z, number_simulations]
            data[i, :] = torch.cat((torch.tensor([w_c]).unsqueeze(1), data_fw[-1, :, :])).squeeze()

    elif params['dataGen'] == 'trajectories':
        # Eigenvalues for D
        w_c = random.uniform(0.5, 2.5) * math.pi
        b, a = signal.bessel(3, w_c, 'low', analog=True, norm='phase')
        eigen = np.roots(a)

        # Place eigenvalue
        observer.D = observer.tensorDFromEigen(eigen)

        # Advance k/min(lambda) in time
        k = 20
        t_c = k/min(abs(observer.eigenD.real))
        tsim = (0, t_c)

        # Create dataframe
        y_0 = torch.zeros((observer.dim_x + observer.dim_z, nsims), dtype=torch.double)

        y_0[:observer.dim_x, :] = torch.transpose(mesh, 0, 1)
        tq, data = observer.simulateLueneberger(y_0, tsim, dt)

        # Bin initial data
        k = 3
        t = 3/min(abs(linalg.eig(observer.D)[0].real))
        idx = max(np.argwhere(tq < t))

        # Data contains the trajectories for every initial value
        # Shape [dim_z, tsim-initial_data, number_simulations]
        data = data[idx[-1]-1:, :, :]

        w_c = torch.tensor([w_c]).unsqueeze(1).repeat(data.shape[0], 1, nsims)

        data = torch.cat((w_c, data), dim=1)

    return data.float()


def processModel(data, observer, model, params):
    timestr = time.strftime("%Y%m%d-%H%M%S")

    path = params['path'] + '/' + timestr

    torch.save(model.state_dict(), path+"_model.pt")

    mesh = generateMesh(params['validation'])
    data_val = generateTrainingData(observer, params['validation'])

    z, x_hat = model(data_val[:, :observer.dim_x+1])

    np.savetxt(path+"_train_data.csv", data, delimiter=",")
    np.savetxt(path+"_val_data.csv", data_val, delimiter=",")
    np.savetxt(path+"_x_hat.csv", x_hat.detach().numpy(), delimiter=",")
    np.savetxt(path+"_z_hat.csv", z.detach().numpy(), delimiter=",")

    with open(path+"_observer.pickle", "wb") as f:
        pickle.dump(observer, f)

    plot.plotLogError2D(mesh, x_hat[:, 1:].detach().numpy(), mesh, params)

    indices = torch.randperm(x_hat[:, 1:].shape[0])[:params['validation']['val_size']]
    y_0_hat = torch.cat((x_hat[indices, 1:], torch.zeros((indices.shape[0], observer.dim_z))), dim=1).T
    y_0 = torch.cat((data_val[indices, 1:observer.dim_x+1], torch.zeros((indices.shape[0], observer.dim_z))), dim=1).T

    tq, w_hat = observer.simulateLueneberger(y_0_hat, (0, 50), params['data']['simulation_step'])
    tq, w = observer.simulateLueneberger(y_0, (0, 50), params['data']['simulation_step'])

    for i in range(len(indices)):
        plot.plotSimulation2D(tq, w[:, :observer.dim_x, i].detach().numpy(),
                              w_hat[:, :observer.dim_x, i].detach().numpy(), params, i)
