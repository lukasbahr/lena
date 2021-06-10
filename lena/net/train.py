from lena.net.autoencoder import Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def trainAutonomousAutoencoder(data, observer, options):

    # Make torch use the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize Autoencoder
    model = Autoencoder(observer, options, device)
    model.to(device)

    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Make use of tensorboard
    if options['is_tensorboard']:
        writer = SummaryWriter()

    # Create trainloader
    trainloader = utils.data.DataLoader(data, batch_size=options['batch_size'],
                                        shuffle=True, num_workers=2, drop_last=True)

    # Train autoencoder
    for epoch in range(options['epochs']):

        # Track loss
        running_loss = 0.0

        # Train
        for i, batch in enumerate(trainloader, 0):
            # Split batch into inputs and labels
            inputs = torch.tensor(batch[:, :observer.dim_x+1]).to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Predict
            z, x_hat = model(inputs)

            # Compute loss
            loss, loss1, loss2 = model.loss(inputs, x_hat, z)

            # Write loss to tensorboard
            if options['is_tensorboard']:
                writer.add_scalars("Loss/train", {
                    'loss': loss,
                    'loss1': loss1,
                    'loss2': loss2,
                }, i + (epoch*len(trainloader)))

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

        print('====> Epoch: {} done!'.format(epoch + 1))

        # Validate prediction after each epoch in tensorboard
        if options['is_tensorboard']:

            randInt = torch.randint(0, data.shape[0], (1,))[0]

            # Predict for a random datapoint
            with torch.no_grad():
                inputs = data[randInt, :observer.dim_x].to(device)
                z, x_hat = model(inputs)

            # Simulation parameters
            tsim = (0, 50)
            dt = 1e-2

            # Set inital simulation value for prediction and truth
            w_0_pred = torch.cat((x_hat.to('cpu'), z.to('cpu'))).reshape(5, 1)
            w_0_truth = data[randInt].reshape(5, 1)

            # Simulate for initial values
            tq, w_pred = observer.simulateLueneberger(w_0_pred, tsim, dt)
            tq_, w_truth = observer.simulateLueneberger(w_0_truth, tsim, dt)

            # Create matplot figure
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(tq, w_pred[:, :2, 0])
            ax.plot(tq_, w_truth[:, :2, 0])

            # Write figure to tensorboard
            writer.add_figure("recon", fig, global_step=epoch, close=True, walltime=None)

    print('Finished Training')

    # Close tensorboard writer
    if options['is_tensorboard']:
        writer.close()

    return model

