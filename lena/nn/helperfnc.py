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


# def generateMesh(params):
#     # Sample either a uniformly grid or use latin hypercube sampling
#     if params['sampling'] == 'uniform':
#         axes = np.arange(params['grid_size'][0], params['grid_size'][1], params['grid_step'])
#         mesh = np.array(np.meshgrid(axes, axes)).T.reshape(-1, 2)
#         mesh = torch.tensor(mesh)
#     elif params['sampling'] == 'lhs':
#         limits = np.array([params['lhs_limits_state'], params['lhs_limits_state']])
#         sampling = LHS(xlimits=limits)
#         mesh = torch.tensor(sampling(params['lhs_samples']))

#     return mesh


# def generateTrainingData(observer, params):
#     """
#     Generate training samples (x,z) by simulating backward in time
#     and after forward in time.
#     """

#     if params['type'] == 'pairs':

#         if params['experiment'] == 'autonomous':
#             mesh = generateMesh(params)
#             nsims = mesh.shape[0]

#             # Data contains (x_i, z_i) pairs in shape [dim_z, number_simulations]
#             data = torch.cat((mesh, torch.zeros((mesh.shape[0], observer.dim_z))), dim=1)

#         elif params['experiment'] == 'noise':

#             limits = np.array([params['lhs_limits_wc'], params['lhs_limits_state'], params['lhs_limits_state']])
#             sampling = LHS(xlimits=limits)
#             data = torch.tensor(sampling(params['lhs_samples']))

#         elif params['experiment'] == 'time':
#             mesh = generateMesh(params)
#             nsims = mesh.shape[0]
#             # y_0 = torch.zeros((observer.dim_x + observer.dim_z, nsims))
#             y_0 = torch.tensor([[0.], [0.], [0], [0], [0]])

#             # Simulate forward in time
#             tsim = (0, 600)
#             # y_0[:observer.dim_x, :] = torch.transpose(mesh, 0, 1)
#             tq, data_fw = observer.simulateSystem(y_0, tsim, params['simulation_step'])

#             data = torch.transpose(data_fw, 0, 1).float()
#             data = data[:,:,0].T

#             # If system is autonomous we may also want to concatenate the timeframe
#             # Copy tq to match data shape
#             # tq = tq.unsqueeze(1).repeat(1, 1, nsims)

#             # Shape [dim_z+1, tsim-initial_data, number_simulations]
#             # data = torch.cat((tq[:,:,0], data[:,:,0])).T

#         return data.float()


def processModel(data, observer, model, params):
    # Get time string
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Set path
    path = params['path'] + '/' + timestr

    # Save torch model
    torch.save(model.state_dict(), path+"_model.pt")

    # Generate validation mesh
    mesh = generateMesh(params['validation'])
    data_val = generateTrainingData(observer, params['validation'])

    # Compute z and x_hat for static test
    z, x_hat = model(data_val[:, :observer.dim_x+observer.optionalDim])
    model_estimate = torch.cat((z, x_hat), dim=1)

    # Plot t-SNE of latent space
    plot.plotTSNE(z.detach().numpy(), True, params['path'])

    # Save data
    np.savetxt(path+"_train_data.csv", data, delimiter=",")
    np.savetxt(path+"_val_data.csv", data_val, delimiter=",")
    np.savetxt(path+"_model_estimate.csv", model_estimate.detach().numpy(), delimiter=",")

    # TODO: Make inter1pd pickable
    # with open(path+"_observer.pickle", "wb") as f:
    # pickle.dump(observer, f)

    # Plot heatmap
    plot.plotLogError2D(mesh, x_hat[:, observer.optionalDim:].detach().numpy(), mesh, params)

    # Simulation parameters
    tsim = params['validation']['tsim']
    dt = params['validation']['dt']
    x_0 = torch.tensor(params['validation']['val_cond'])

    # Get measurements y by simulating from $x$ forward in time
    w_0 = torch.cat((x_0, torch.zeros((x_0.shape[0], observer.dim_z))), dim=1).T
    tq_, w_truth = observer.simulateSystem(w_0, tsim, dt)
    x_truth = w_truth[:, observer.optionalDim:observer.optionalDim+observer.dim_x, :]

    for i in range(x_0.shape[0]):
        # Solve $z_dot$
        y = torch.cat((tq_.unsqueeze(1), observer.h(x_truth[:, :, i].T).T), dim=1)
        tq_pred, w_pred = observer.simulateLueneberger(y, tsim, dt)

        # Predict $x_hat$ with $T_star(z)$
        with torch.no_grad():
            x_hat = model.decoder(w_pred[:, :, 0].float())

        plot.plotObserverEstimation2D(tq_pred, x_truth[:, :, i], x_hat, True, params['path'])

        sim_data = torch.cat((tq_.unsqueeze(1),  x_truth[:, :, i], x_hat), dim=1)
        condstr = str(x_truth[0, 0, i]) + "_" + str(x_truth[0, 1, i])

        np.savetxt(path+"_train_data.csv", sim_data, delimiter=",", header="tq,x_1,x_2,x_hat_1,x_hat_2")
