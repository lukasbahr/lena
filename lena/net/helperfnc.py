import numpy as np
import torch
from scipy import linalg
from smt.sampling_methods import LHS


def normalize(self, data):
    """
    Normalize data between [0,1]
    """
    return (data-np.min(data))/(np.max(data) - np.min(data))


def generateTrainingData(observer, options):
    """
    Generate training samples (x,z) by simulating backward in time
    and after forward in time.
    """

    # Sample either a uniformly grid or use latin hypercube sampling
    if options['sampling'] == 'uniform':
        mesh = np.array(np.meshgrid(options['gridSize'], options['gridSize'])).T.reshape(-1, 2)
        mesh = torch.tensor(mesh)
    elif options['sampling'] == 'lhs':
        sampling = LHS(xlimits=options['lhs_limits'])
        mesh = torch.tensor(sampling(options['lhs_samples']))

    # Advance k/min(lambda) in time
    k = 10
    t_c = k/min(abs(observer.eigenD.real))
    nsims = mesh.shape[0]

    # Set simulation step width
    dt = options['simulationStep']

    # Generate either pairs of (x_i, z_i) values by simulating back and then forward in time
    # or generate trajectories for every initial value (x_1_i, x_2_i, z_0) 
    if options['dataGen'] == 'pairs':

        # Create dataframes
        y_0 = torch.zeros((observer.dim_x + observer.dim_z, nsims), dtype=torch.double)
        y_1 = torch.zeros((observer.dim_x + observer.dim_z, nsims), dtype=torch.double)

        # Simulate backward in time
        tsim = (0, -t_c)
        y_0[:observer.dim_x, :] = torch.transpose(mesh, 0, 1)
        tq, data_bw = observer.simulateLueneberger(y_0, tsim, -dt)

        # Simulate forward in time starting from the last point from previous simulation
        tsim = (-t_c, 0)
        y_1[:observer.dim_x, :] = data_bw[-1, :observer.dim_x, :]
        tq, data_fw = observer.simulateLueneberger(y_0, tsim, dt)

        # Data contains (x_i, z_i) pairs in shape [dim_z, number_simulations]
        data = torch.transpose(data_fw[-1, :, :], 0, 1).float()

    elif options['dataGen'] == 'trajectories':
        # Create dataframe
        y_0 = torch.zeros((observer.dim_x + observer.dim_z, nsims), dtype=torch.double)

        # Simulate forward in time
        tsim = (0, t_c)
        y_0[:observer.dim_x, :] = torch.transpose(mesh, 0, 1)
        tq, data_fw = observer.simulateLueneberger(y_0, tsim, dt)

        data = torch.transpose(data_fw, 0, 1).float()

        # Bin initial data
        k=3
        t_c = 3/min(abs(linalg.eig(observer.D)[0].real))
        idx = max(np.argwhere(tq < t_c))

        # Data contains the trajectories for every initial value
        # Shape [dim_z, tsim-initial_data, number_simulations]
        data = data[:,idx[-1]-1:,:]
        tq = tq[idx[-1]-1:]

        # If system is autonomous we may also want to concatenate the timeframe
        if options['isAutonomous']:
            # Copy tq to match data shape
            tq = tq.unsqueeze(1).repeat(1, 1, nsims)

            # Shape [dim_z+1, tsim-initial_data, number_simulations]
            data = torch.cat((tq, data))

    return data
