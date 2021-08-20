from lena.nn.autoencoder import Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def train_nn(data, observer, params):
    dim_x = observer.dim_x
    dim_z = observer.dim_z
    optionalDim = observer.optionalDim

    # Make torch use the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize Autoencoder
    model = NN(observer, params, device)
    # model = Autoencoder(observer, params, device)
    model.to(device)

    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['lr_milestones'], gamma=0.1)

    # Make use of tensorboard
    if params['is_tensorboard']:
        writer = SummaryWriter(params['tensoboard_path'])

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
            x = batch.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Predict
            x_hat = model(z)

            loss = model.loss(x, x_hat)

            # Write loss to tensorboard
            if params['is_tensorboard']:
                writer.add_scalars("Loss/train", {
                    'loss': loss,
                }, i + (epoch*len(trainloader)))
                writer.flush()

            # Gradient step and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

            # Print every 200 mini batches
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.00

        print('====> Epoch: {} done! LR: {}'.format(epoch + 1, optimizer.param_groups[0]["lr"]))
        print('x:{}, x_hat:{}'.format(x, x_hat))

        # Adjust learning rate
        scheduler.step()

    print('Finished Training')

    # Close tensorboard writer
    if params['is_tensorboard']:
        writer.close()

    return model


def train(data, observer, params):
    dim_x = observer.dim_x
    dim_z = observer.dim_z
    optionalDim = observer.optionalDim

    # Make torch use the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize Autoencoder
    model = Autoencoder(observer, params, device)
    # model = Autoencoder(observer, params, device)
    model.to(device)

    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['lr_milestones'], gamma=0.1)

    # Make use of tensorboard
    if params['is_tensorboard']:
        writer = SummaryWriter(params['tensoboard_path'])

    # Create trainloader
    trainloader = utils.data.DataLoader(data, batch_size=params['batch_size'],
                                        shuffle=params['shuffle'], num_workers=2, drop_last=True)

    # Train autoencoder
    for epoch in range(params['epochs']):

        # Track loss
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0

        # Train
        for i, batch in enumerate(trainloader, 0):
            # Split batch into inputs and labels

            x = batch.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Predict
            z_hat, x_hat = model(x)


            # Compute loss
            if params['experiment'] == 'autonomous':
                loss, loss1, loss2 = model.loss_auto(x, x_hat, z_hat)
            elif params['experiment'] == 'noise':
                loss, loss1, loss2 = model.loss_noise(x, x_hat, z_hat)
            elif params['experiment'] == 'time':
                loss, loss1, loss2 = model.loss_time(x, x_hat, z_hat)

            # Write loss to tensorboard
            if params['is_tensorboard']:
                writer.add_scalars("Loss/train", {
                    'loss': loss,
                    'loss1': loss1,
                    'loss2': loss2,
                }, i + (epoch*len(trainloader)))
                writer.flush()

            # Gradient step and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()

            # Print every 200 mini batches
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f loss_1: %.3f loss_2: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200, running_loss1/200, running_loss2/200))
                running_loss = 0.00
                running_loss1 = 0.00
                running_loss2 = 0.00

        print('====> Epoch: {} done! LR: {}'.format(epoch + 1, optimizer.param_groups[0]["lr"]))
        print('x:{}, x_hat:{}'.format(x, x_hat))

        # Adjust learning rate
        scheduler.step()

        # # Validate prediction after every second epoch in tensorboard
        # if params['is_tensorboard'] and epoch % 4 == 0:

        #     # Simulation parameters
        #     tsim = (0, 100)
        #     dt = 1e-2

        #     # Get measurements fom x forward in time
        #     w_0_truth = torch.tensor([[1., 2., 0., 0., 0.]]).T
        #     observer.u = controller.lin_chirp_controller
        #     tq_, w_truth = observer.simulateSystem(w_0_truth, tsim, dt)

        #     # Solve z_dot with measurement y
        #     y = torch.cat((tq_.unsqueeze(1), observer.h(
        #         w_truth[:, :observer.dim_x, 0].T).T), dim=1)
        #     tq_pred, w_pred = observer.simulateLueneberger(y, tsim, dt)
        #     w_pred = w_pred[:, :, 0]

        #     if params['experiment'] == 'noise':
        #         w_pred = torch.cat((torch.tensor([1.0]).repeat(w_pred.shape[0], 1), w_pred[:, :]), dim=1)
        #     elif params['experiment'] == 'time':
        #         tq_pred = tq_pred/torch.max(tq_pred)
        #         w_pred = torch.cat((tq_pred.unsqueeze(1), w_pred),1)

        #     # Predict x_hat with T_star(z)
        #     with torch.no_grad():
        #         x_hat = model.decoder(w_pred.float())

        #     x_hat = x_hat[:,1:]
        #     # Create fig with 300 dpi
        #     fig = plt.figure(dpi=200)

        #     # Create ax_trans figure
        #     ax_x1 = fig.add_subplot(4, 1, 1)
        #     ax_x1.plot(tq_, x_hat[:, 0].to("cpu"), color='red', linestyle='dashed', label='x_hat')
        #     ax_x1.plot(tq_, w_truth[:, 0, 0], color='blue', label='x')

        #     ax_x1.set_ylabel(r'$x_1$')
        #     ax_x1.set_xlabel('time' + r'$[s]$')

        #     # Create ax_trans figure
        #     ax_x2 = fig.add_subplot(4, 1, 2)
        #     ax_x2.plot(tq_, x_hat[:, 1].to("cpu"), color='red', linestyle='dashed', label='x_hat')
        #     ax_x2.plot(tq_, w_truth[:, 1, 0], color='blue', label='x')

        #     ax_x2.set_ylabel(r'$x_2$')
        #     ax_x2.set_xlabel('time' + r'$[s]$')

        #     # Create ax_trans figure
        #     ax_error = fig.add_subplot(4, 1, 3)
        #     ax_error.plot(tq_, sqrtError(x_hat[:, 0], w_truth[:, 0, 0]), color='red', label='x_hat')
        #     ax_error.plot(tq_, sqrtError(x_hat[:, 1], w_truth[:, 1, 0]), color='blue', label='x')

        #     ax_error.set_ylabel(r'$x_i-\hat{x_i}$')
        #     ax_error.set_xlabel('time' + r'$[s]$')

        #     # Create ax_z figure
        #     ax_z = fig.add_subplot(4, 1, 4)
        #     ax_z.plot(tq_pred, w_pred[:, observer.optionalDim:observer.optionalDim+observer.dim_z])

        #     ax_z.set_ylabel(r'$z_i$')
        #     ax_z.set_xlabel('time' + r'$[s]$')

        #     fig.tight_layout()

        #     # Write figure to tensorboard
        #     writer.add_figure("recon", fig, global_step=epoch, close=True, walltime=None)
        #     writer.flush()

        # # Validate prediction after every second epoch in tensorboard
        # if params['is_tensorboard'] and epoch % 4 == 0:

        #     # Simulation parameters
        #     tsim = (0, 100)
        #     dt = 1e-2

        #     # Get measurements fom x forward in time
        #     w_0_truth = torch.tensor([[1., 2., 0., 0., 0.]]).T
        #     observer.u = controller.sin_controller
        #     tq_, w_truth = observer.simulateSystem(w_0_truth, tsim, dt)

        #     # Solve z_dot with measurement y
        #     y = torch.cat((tq_.unsqueeze(1), observer.h(
        #         w_truth[:, :observer.dim_x, 0].T).T), dim=1)
        #     tq_pred, w_pred = observer.simulateLueneberger(
        #         y, tsim, dt, model=model, u_0=controller.lin_chirp_controller)
        #     w_pred = w_pred[:, :, 0]

        #     if params['experiment'] == 'noise':
        #         w_pred = torch.cat((torch.tensor([1.0]).repeat(w_pred.shape[0], 1), w_pred[:, :]), dim=1)
        #     elif params['experiment'] == 'time':
        #         tq_pred = tq_pred/torch.max(tq_pred)
        #         w_pred = torch.cat((tq_pred.unsqueeze(1), w_pred),1)

        #     # Predict x_hat with T_star(z)
        #     with torch.no_grad():
        #         x_hat = model.decoder(w_pred.float())

        #     x_hat = x_hat[:,0:]
        #     # Create fig with( 300 dpi
        #     fig = plt.figure(dpi=200)

        #     # Create ax_trans figure
        #     ax_x1 = fig.add_subplot(4, 1, 1)
        #     ax_x1.plot(tq_, x_hat[:, 0].to("cpu"), color='red', linestyle='dashed', label='x_hat')
        #     ax_x1.plot(tq_, w_truth[:, 0, 0], color='blue', label='x')

        #     ax_x1.set_ylabel(r'$x_1$')
        #     ax_x1.set_xlabel('time' + r'$[s]$')

        #     # Create ax_trans figure
        #     ax_x2 = fig.add_subplot(4, 1, 2)
        #     ax_x2.plot(tq_, x_hat[:, 1].to("cpu"), color='red', linestyle='dashed', label='x_hat')
        #     ax_x2.plot(tq_, w_truth[:, 1, 0], color='blue', label='x')

        #     ax_x2.set_ylabel(r'$x_2$')
        #     ax_x2.set_xlabel('time' + r'$[s]$')

        #     # Create ax_trans figure
        #     ax_error = fig.add_subplot(4, 1, 3)
        #     ax_error.plot(tq_, sqrtError(x_hat[:, 0], w_truth[:, 0, 0]), color='red', label='x_hat')
        #     ax_error.plot(tq_, sqrtError(x_hat[:, 1], w_truth[:, 1, 0]), color='blue', label='x')

        #     ax_error.set_ylabel(r'$x_i-\hat{x_i}$')
        #     ax_error.set_xlabel('time' + r'$[s]$')

        #     # Create ax_z figure
        #     ax_z = fig.add_subplot(4, 1, 4)
        #     ax_z.plot(tq_pred, w_pred[:, observer.optionalDim:observer.optionalDim+observer.dim_z])

        #     ax_z.set_ylabel(r'$z_i$')
        #     ax_z.set_xlabel('time' + r'$[s]$')

        #     fig.tight_layout()

        #     # Write figure to tensorboard
        #     writer.add_figure("recon with different u", fig, global_step=epoch, close=True, walltime=None)
        #     writer.flush()

    print('Finished Training')

    # Close tensorboard writer
    if params['is_tensorboard']:
        writer.close()

    return model

def sqrtError(x, x_hat):
    return x-x_hat
