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

        if  params['experiment'] == 'autonomous':

            # Data contains (x_i, z_i) pairs in shape [dim_z, number_simulations]
            data = torch.cat((mesh, torch.zeros((mesh.shape[0], observer.dim_z))),dim=1)

        elif params['experiment'] == 'noise':
            # Create dataframe
            data = torch.zeros((nsims, observer.dim_x + observer.dim_z + observer.optionalDim))

            for i in range(nsims):
                # Eigenvalues for D
                w_c = random.uniform(2.5, 2.5) * math.pi
                b, a = signal.bessel(3, w_c, 'low', analog=True, norm='phase')
    
                eigen = np.roots(a)
    
                # Place eigenvalue
                observer.D = observer.tensorDFromEigen(eigen)
    
                # Data contains (x_i, z_i, w_c_i) pairs in shape [1+dim_x+dim_z, number_simulations]
                w_0 = torch.cat((mesh[i,:], torch.zeros((observer.dim_z))))
                data[i, :] = torch.cat((torch.tensor([w_c]), w_0))

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
