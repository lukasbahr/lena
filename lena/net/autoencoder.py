from lena.observer.lueneberger import LuenebergerObserver
import torch
from torch import nn
from scipy import linalg, signal
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, observer: LuenebergerObserver, params, device):
        """
        Autoencoder(observer, params) - class for learning nonlinear transformations
        """
        super(Autoencoder, self).__init__()

        # Namespace observer, params and device
        self.observer = observer
        self.params = params
        self.device = device

        dim_x = observer.dim_x + observer.optionalDim
        dim_z = observer.dim_z + observer.optionalDim

        numHL = params['num_hlayers']
        sizeHL = params['size_hlayers']

        # Set activation function
        if params['activation'] == "relu":
            self.act = nn.ReLU()
        elif params['activation'] == "tanh":
            self.act = nn.Tanh()

        # Encoder architecture
        self.encoderLayers = nn.ModuleList()
        self.encoderLayers.append(nn.Linear(dim_x, sizeHL))
        for i in range(numHL):
            self.encoderLayers.append(self.act)
            self.encoderLayers.append(nn.Linear(sizeHL, sizeHL))
        self.encoderLayers.append(nn.Linear(sizeHL, dim_z))

        # Decoder architecture
        self.decoderLayers = nn.ModuleList()
        self.decoderLayers.append(nn.Linear(dim_z, sizeHL))
        for i in range(numHL):
            self.decoderLayers.append(self.act)
            self.decoderLayers.append(nn.Linear(sizeHL, sizeHL))
        self.decoderLayers.append(nn.Linear(sizeHL, dim_x))

    def encoder(self, x):
        """
        Encode input to latent space.
        """
        for layer in self.encoderLayers:
            x = layer(x)
        return x

    def decoder(self, x):
        """
        Decode latent space and reconstruct input x
        """
        for layer in self.decoderLayers:
            x = layer(x)
        return x

    def loss_auto(self, x, x_hat, z_hat):
        """
        Loss function for autonomous experiment
        """
        mse = nn.MSELoss()

        # Compute gradients of T_u with respect to inputs
        dTdx = torch.autograd.functional.jacobian(
            self.encoder, x, create_graph=False, strict=False, vectorize=False)
        dTdx = dTdx[dTdx != 0].reshape((self.params['batch_size'], self.observer.dim_z, self.observer.dim_x))

        loss1 = self.params['recon_lambda'] * mse(x, x_hat)

        lhs = torch.zeros((self.observer.dim_z, self.params['batch_size']))
        for i in range(self.params['batch_size']):
            lhs[:, i] = torch.matmul(dTdx[i], self.observer.f(x.T).T[i]).T

        rhs = torch.matmul(self.observer.D.to(self.device),
                           z_hat.T) + torch.matmul(self.observer.F.to(self.device),
                                               self.observer.h(x.T).to(self.device))

        loss2 = mse(lhs.to(self.device), rhs)

        loss = loss1 + loss2 

        return loss, loss1, loss2

    def loss_noise(self, y, y_pred, latent):
        """
        Loss function for noise experiment
        """
        w_c = y[:, 0]

        x = y[:, 1:]
        z = latent[:, 1:]

        mse = nn.MSELoss()

        loss1 = self.params['recon_lambda'] * mse(y, y_pred)

        # Compute gradients of T_u with respect to inputs
        dTdy = torch.autograd.functional.jacobian(self.encoder, y)
        dTdy = dTdy[dTdy != 0].reshape((self.params['batch_size'], self.observer.dim_z+1, self.observer.dim_x+1))
        dTdx = dTdy[:, 1:, 1:]

        lhs = torch.zeros((self.observer.dim_z, self.params['batch_size']))
        rhs = lhs.clone() 

        for i in range(self.params['batch_size']):
            b, a = signal.bessel(3, w_c[i], 'low', analog=True, norm='phase')
            eigen = np.roots(a)

            # Place eigenvalue
            D = self.observer.tensorDFromEigen(eigen).to(self.device)

            lhs[:, i] = torch.matmul(dTdx[i], self.observer.f(x.T).T[i]).T
            rhs[:, i] = (torch.matmul(D, z[i].T).reshape(-1, 1) + torch.matmul(self.observer.F.to(self.device),
                         self.observer.h(x[i].reshape(-1, 1))).to(self.device)).squeeze()

        loss2 = mse(lhs.to(self.device), rhs)


        loss = loss1 + loss2 

        return loss, loss1, loss2


    def forward(self, x):
        """
        Takes a batch of samples, encodes them, and then decodes them.
        """
        # Compute latent space
        z = self.encoder(x)

        # Decode laten space
        x_hat = self.decoder(z)

        return z, x_hat
