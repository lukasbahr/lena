from lena.observer.lueneberger import LuenebergerObserver
import torch
from torch import nn
from scipy import linalg, signal
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, observer: LuenebergerObserver, options, device):
        super(Autoencoder, self).__init__()
        self.observer = observer
        self.options = options
        self.device = device

        numHL = options['numHiddenLayers']
        sizeHL = options['sizeHiddenLayer']

        if options['activation'] == "relu":
            self.act = nn.ReLU()
        elif options['activation'] == "tanh":
            self.act = nn.Tanh()
        else:
            print(
                "Activation function {} not found. Available options: ['relu', 'tanh'].".format(
                    options['activation']))

        # Encoder architecture
        self.encoderLayers = nn.ModuleList()
        self.encoderLayers.append(nn.Linear(self.observer.dim_x + 1, sizeHL))
        for i in range(numHL):
            self.encoderLayers.append(nn.Linear(sizeHL, sizeHL))
        self.encoderLayers.append(nn.Linear(sizeHL, self.observer.dim_z + 1))

        # Decoder architecture
        self.decoderLayers = nn.ModuleList()
        self.decoderLayers.append(nn.Linear(self.observer.dim_z + 1, sizeHL))
        for i in range(numHL):
            self.decoderLayers.append(nn.Linear(sizeHL, sizeHL))
        self.decoderLayers.append(nn.Linear(sizeHL, self.observer.dim_x + 1))

    def encoder(self, x):
        """Encode a batch of samples, and return posterior parameters for each point."""
        for layer in self.encoderLayers:
            x = self.act(layer(x))
        return x

    def decoder(self, x):
        for layer in self.decoderLayers:
            x = self.act(layer(x))
        return x

    def loss(self, y, y_pred, latent):
        w_c = y[:, 0]
        x = y[:, 1:]
        z = latent[:, 1:]

        mse = nn.MSELoss()

        loss1 = self.options['reconLambda'] * mse(y, y_pred)

        # Compute gradients of T_u with respect to inputs
        dTdy = torch.autograd.functional.jacobian(self.encoder, y)
        dTdy = dTdy[dTdy != 0].reshape((self.options['batchSize'], self.observer.dim_z+1, self.observer.dim_x+1))
        dTdx = dTdy[:, 1:, 1:]

        lhs = torch.zeros((self.observer.dim_z, self.options['batchSize']))
        rhs = torch.zeros((self.observer.dim_z, self.options['batchSize']))

        for i in range(self.options['batchSize']):
            b, a = signal.bessel(3, w_c[i], 'low', analog=True, norm='phase')
            eigen = np.roots(a)

            # Place eigenvalue
            D = self.observer.tensorDFromEigen(eigen).to(self.device)

            lhs[:, i] = torch.matmul(dTdx[i], self.observer.f(x.T).T[i]).T
            rhs[:, i] = (torch.matmul(D, z.T[i]).reshape(-1, 1) + torch.matmul(self.observer.F.to(self.device),
                         self.observer.h(x[i].reshape(-1, 1))).to(self.device)).squeeze()

        loss2 = mse(lhs.to(self.device), rhs)

        loss = loss1 + loss2

        return loss, loss1, loss2

    def forward(self, x):
        """Takes a batch of samples, encodes them, and then decodes them again to compare."""

        z = self.encoder(x)

        x_hat = self.decoder(z)

        return z, x_hat
