from lena.net.autoencoder import Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


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
                                        shuffle=True, num_workers=2, drop_last=True)

    # Train autoencoder
    for epoch in range(params['epochs']):

        # Track loss
        running_loss = 0.0

        # Train
        for i, batch in enumerate(trainloader, 0):
            # Split batch into inputs and labels
            inputs = batch[:, :dim_x+optionalDim].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Predict
            z, x_hat = model(inputs)

            # Compute loss
            if params['experiment'] == 'autonomous':
                loss, loss1, loss2 = model.loss_auto(inputs, x_hat, z)
            elif params['experiment'] == 'noise':
                loss, loss1, loss2 = model.loss_noise(inputs, x_hat, z)

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

            # Print every 200 mini batches
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.00

        print('====> Epoch: {} done! LR: {}'.format(epoch + 1, optimizer.param_groups[0]["lr"]))

        # Adjust learning rate
        scheduler.step()

        # Validate random prediction after each epoch in tensorboard
        if params['is_tensorboard']:

            # Predict for a random datapoint
            randInt = torch.randint(0, data.shape[0], (1,))[0]

            # Simulation parameters
            tsim = (0, 40)
            dt = 1e-2

            # Get measurements y
            w_0_truth = torch.tensor([[0.4, 0.4, 0, 0, 0]]).T
            tq_, w_truth = observer.simulateLueneberger(w_0_truth, tsim, dt)

            # Solve $z_dot$
            y = torch.cat((tq_.unsqueeze(1), observer.h(
                w_truth[:, observer.optionalDim:observer.optionalDim+observer.dim_x, 0].T).T), dim=1)
            tq_pred, w_pred = observer.simulateZ(y, tsim, dt)

            # Predict x_hat with T_star(z_i)
            with torch.no_grad():
                x_hat = model.decoder(w_pred[:, :, 0].float())

            # Create fig with 300 dpi
            fig = plt.figure(dpi=50)
            fig.tight_layout(pad=3.0)

            # Create ax_trans figure
            ax_x1 = fig.add_subplot(1, 3, 1)
            ax_x1.plot(tq_, x_hat.to("cpu"), color='red', linestyle='dashed', label='x_hat')
            ax_x1.plot(tq_, w_truth[:, 0, 0], color='blue', label='x')

            ax_x1.set_ylabel('state')
            ax_x1.set_xlabel('time')
            ax_x1.set_title('Simulation x_1, x_hat_1')

            # Create ax_trans figure
            ax_x2 = fig.add_subplot(1, 3, 2)
            ax_x2.plot(tq_, x_hat.to("cpu"), color='red', linestyle='dashed', label='x_hat')
            ax_x2.plot(tq_, w_truth[:, 1, 0], color='blue', label='x')

            ax_x2.set_ylabel('state')
            ax_x2.set_xlabel('time')
            ax_x2.set_title('Simulation x_2, x_hat_2')

            # Create ax_z figure
            ax_z = fig.add_subplot(1, 3, 3)
            ax_z.plot(tq_pred, w_pred[:, :, 0])

            ax_z.set_ylabel('state')
            ax_z.set_xlabel('time')
            ax_z.set_title('Simulation z')

            # Write figure to tensorboard
            writer.add_figure("recon", fig, global_step=epoch, close=True, walltime=None)
            writer.flush()

    print('Finished Training')

    # Close tensorboard writer
    if params['is_tensorboard']:
        writer.close()

    return model
