from lena.net.autoencoder import Autoencoder
from lena.net.linear import Model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def trainNonlinearLuenbergerTransformation(
        data: torch.Tensor, observer, isForwardTrans: bool, params):
    """
    Numerically estimate the
    nonlinear Luenberger transformation of a SISO input-affine nonlinear
    system with static transformation, and the corresponding left-inverse.
    """
    dim_x = observer.dim_x
    dim_z = observer.dim_z

    # Set size according to compute either T or T*
    if isForwardTrans:
        netSize = (dim_x, dim_z)
        dataInput = (0, dim_x)
        dataOutput = (dim_x, dim_x+dim_z)
    else:
        netSize = (dim_z, dim_x)
        dataInput = (dim_x, dim_x+dim_z)
        dataOutput = (0, dim_x)

    # Make use of tensorboard
    if params['is_tensorboard']:
        writer = SummaryWriter()

    # Make torch use the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model params
    model = Model(netSize[0], netSize[1])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    # Network params
    criterion = nn.MSELoss()

    # Create trainloader
    trainloader = utils.data.DataLoader(data, batch_size=params['batch_size'],
                                        shuffle=params['shuffle'], num_workers=2)

    # Train Transformation
    # Loop over dataset
    for epoch in range(params['epochs']):

        # Track loss
        running_loss = 0.0

        # Train
        for i, data in enumerate(trainloader, 0):
            # Set input and labels
            inputs = data[:, dataInput[0]:dataInput[1]].to(device)
            labels = data[:, dataOutput[0]:dataOutput[1]].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
                            # Write loss to tensorboard
            if params['is_tensorboard']:
                writer.add_scalars("Loss/train", {
                    'loss': loss,
                    }, i + (epoch*len(trainloader)))
                writer.flush()

            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.00

        print('====> Epoch: {} done!'.format(epoch + 1))

        # Validate random prediction after each epoch in tensorboard
        if params['is_tensorboard']:

            # Simulation parameters
            tsim = (0, 100)
            dt = 1e-2

            # Get measurements y by simulating from $x$ forward in time
            w_0_truth = torch.tensor([[1,2, 0., 0., 0.]]).T
            tq_, w_truth = observer.simulateSystem(w_0_truth, tsim, dt)

            # Solve $z_dot$
            y = torch.cat((tq_.unsqueeze(1), observer.h(
                w_truth[:, observer.optionalDim:observer.optionalDim+observer.dim_x, 0].T).T), dim=1)
            tq_pred, w_pred = observer.simulateLueneberger(y, tsim, dt)

            # Predict $x_hat$ with $T_star(z)$
            with torch.no_grad():
                x_hat = model(w_pred[:, :, 0].float())

            # Create fig with 300 dpi
            fig = plt.figure(dpi=200)

            # Create ax_trans figure
            ax_x1 = fig.add_subplot(3, 1, 1)
            ax_x1.plot(tq_, x_hat[:,0].to("cpu"), color='red', linestyle='dashed', label='x_hat')
            ax_x1.plot(tq_, w_truth[:, 0, 0], color='blue', label='x')

            ax_x1.set_ylabel(r'$x_1$')
            ax_x1.set_xlabel('time' + r'$[s]$')

            # Create ax_trans figure
            ax_x2 = fig.add_subplot(3, 1, 2)
            ax_x2.plot(tq_, x_hat[:,1].to("cpu"), color='red', linestyle='dashed', label='x_hat')
            ax_x2.plot(tq_, w_truth[:, 1, 0], color='blue', label='x')

            ax_x2.set_ylabel(r'$x_2$')
            ax_x2.set_xlabel('time' + r'$[s]$')

            # Create ax_z figure
            ax_z = fig.add_subplot(3, 1, 3)
            ax_z.plot(tq_pred, w_pred[:, :, 0])

            ax_z.set_ylabel(r'$z_i$')
            ax_z.set_xlabel('time' + r'$[s]$')

            fig.tight_layout()

            # Write figure to tensorboard
            writer.add_figure("recon", fig, global_step=epoch, close=True, walltime=None)
            writer.flush()

    print('Finished Training')

    return model


def train(data, observer, params):
    dim_x = observer.dim_x
    dim_z = observer.dim_z
    optionalDim = observer.optionalDim

    # Make torch use the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize Autoencoder
    model = Autoencoder(observer, params, device)
    model.to(device)

    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['lr_milestones'], gamma=0.1)

    # Make use of tensorboard
    if params['is_tensorboard']:
        writer = SummaryWriter()

    # Create trainloader
    trainloader = utils.data.DataLoader(data, batch_size=params['batch_size'],
                                        shuffle=params['shuffle'], num_workers=2, drop_last=True)
    # Train autoencoder
    for epoch in range(params['epochs']):

        # Track loss
        running_loss = 0.0

        # Train
        for i, batch in enumerate(trainloader, 0):
            # Split batch into inputs and labels
            x = batch[:, :dim_x+optionalDim].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Predict
            z_hat, x_hat = model(x)

            # Compute loss
            if params['experiment'] == 'autonomous':
                loss, loss1, loss2 = model.loss_auto(x, x_hat, z_hat)
            elif params['experiment'] == 'noise':
                loss, loss1, loss2 = model.loss_noise(x, x_hat, z_hat)

            # Write loss to tensorboard
            if params['is_tensorboard']:
                writer.add_scalars("Loss/train", {
                    'loss': loss,
                    'loss1': loss1,
                    'loss2': loss2,
                }, i +  (epoch*len(trainloader)))
                writer.flush()

            # Gradient step and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

            # Print every 200 mini batches
            print('[%d] loss: %.3f, loss1: %.3f, loss2: %.3f' %
                    (epoch + 1, loss, loss1, loss2))
            running_loss = 0.00

        print('====> Epoch: {} done! LR: {}'.format(epoch + 1, optimizer.param_groups[0]["lr"]))

        # Adjust learning rate
        scheduler.step()

        # Validate random prediction after each epoch in tensorboard
        if params['is_tensorboard']:

            # Simulation parameters
            tsim = (0, 100)
            dt = 1e-2

            # Get measurements y by simulating from $x$ forward in time
            w_0_truth = torch.tensor([[1.2,2.2, 0., 0., 0.]]).T
            tq_, w_truth = observer.simulateSystem(w_0_truth, tsim, dt)

            # Solve $z_dot$
            y = torch.cat((tq_.unsqueeze(1), observer.h(
                w_truth[:, observer.optionalDim:observer.optionalDim+observer.dim_x, 0].T).T), dim=1)
            tq_pred, w_pred = observer.simulateLueneberger(y, tsim, dt)

            # Predict $x_hat$ with $T_star(z)$
            with torch.no_grad():
                x_hat = model.decoder(w_pred[:, :, 0].float())


            # Create fig with 300 dpi
            fig = plt.figure(dpi=200)

            # Create ax_trans figure
            ax_x1 = fig.add_subplot(3, 1, 1)
            ax_x1.plot(tq_, x_hat[:,0].to("cpu"), color='red', linestyle='dashed', label='x_hat')
            ax_x1.plot(tq_, w_truth[:, 0, 0], color='blue', label='x')

            ax_x1.set_ylabel(r'$x_1$')
            ax_x1.set_xlabel('time' + r'$[s]$')

            # Create ax_trans figure
            ax_x2 = fig.add_subplot(3, 1, 2)
            ax_x2.plot(tq_, x_hat[:,1].to("cpu"), color='red', linestyle='dashed', label='x_hat')
            ax_x2.plot(tq_, w_truth[:, 1, 0], color='blue', label='x')

            ax_x2.set_ylabel(r'$x_2$')
            ax_x2.set_xlabel('time' + r'$[s]$')

            # Create ax_z figure
            ax_z = fig.add_subplot(3, 1, 3)
            ax_z.plot(tq_pred, w_pred[:, :, 0])

            ax_z.set_ylabel(r'$z_i$')
            ax_z.set_xlabel('time' + r'$[s]$')

            fig.tight_layout()

            # Write figure to tensorboard
            writer.add_figure("recon", fig, global_step=epoch, close=True, walltime=None)
            writer.flush()

    print('Finished Training')

    # Close tensorboard writer
    if params['is_tensorboard']:
        writer.close()

    return model
