from lena.observer.lueneberger import LuenebergerObserver
import torch
from torch import nn
from scipy import linalg, signal
import math
import numpy as np


class Autoencoder(nn.Module):
    """
    Class for learning a nonlinear transformation function T_star.
    Use either loss_auto for training autonomous experiments or
    loss_noise for learning eigenvalues.
    """

    def __init__(self, observer: LuenebergerObserver, params, device):
        """Constructs autoencoder instance.

        Arguments:
            observer: An instance of the observer we want to train our
                transformation function for.
            parms: A list of model options like epoch, or batch size to configure
                the autoencoder.

        List of params:
            batch_size: int
            epochs: int
            num_hlayers: int
            size_hlayers: int
            activation: string (tanh, relu)
            lr: float
            lr_milestones: array[float]
            lr_gamma: float
            recon_lambda: float
            is_tensorboar: bool
            shuffle: bool

        Example:
            model = Autoencoder(observer, params)
            z, x_hat = model(data)
            loss, loss_1, loss_2 = loss_auto(x, x_hat, z_hat)
        """
        # Init super class
        super(Autoencoder, self).__init__()

        # Namespace observer, params and device
        self.observer = observer
        self.params = params
        self.device = device

        # Set x and z dimension
        dim_x = observer.dim_x + observer.optionalDim
        dim_z = observer.dim_z + observer.optionalDim

        # Set model params
        numHL = params['num_hlayers']
        sizeHL = params['size_hlayers']

        # Set activation function
        if params['activation'] == "relu":
            self.act = nn.ReLU()
        elif params['activation'] == "tanh":
            self.act = nn.Tanh()

        # Define encoder architecture
        self.encoderLayers = nn.ModuleList()
        self.encoderLayers.append(nn.Linear(dim_x, sizeHL))
        for i in range(numHL):
            self.encoderLayers.append(self.act)
            self.encoderLayers.append(nn.Linear(sizeHL, sizeHL))
        self.encoderLayers.append(nn.Linear(sizeHL, dim_z))

        # Define decoder architecture
        self.decoderLayers = nn.ModuleList()
        self.decoderLayers.append(nn.Linear(dim_z, sizeHL))
        for i in range(numHL):
            self.decoderLayers.append(self.act)
            self.decoderLayers.append(nn.Linear(sizeHL, sizeHL))
        self.decoderLayers.append(nn.Linear(sizeHL, dim_x))

    def encoder(self, x: torch.tensor) -> torch.tensor:
        """
        Encodes input data and returns latent space data.

        Arguments:
            x: Input data of shape [input_dim, dim_x+optionalDim].

        Returns:
            Latent space data [input_dim, dim_z+optionalDim].
        """
        for layer in self.encoderLayers:
            x = layer(x)
        return x

    def decoder(self, z) -> torch.tensor:
        """
        Decodes llatent space and estimates x_hat.

        Arguments:
            z: Input data of shape [input_dim, dim_z+optionalDim].

        Returns:
            Estimate of x_hat [input_dim, dim_x+optionalDim ]
        """
        for layer in self.decoderLayers:
            z = layer(z)
        return z

    def loss_auto(
            self, x: torch.tensor, x_hat: torch.tensor, z_hat: torch.tensor) -> [
            torch.tensor, torch.tensor, torch.tensor]:
        """
        Loss function for autonomous experiment.

        Arguments:
            x: Input data of shape [input_dim, dim_x+optionalDim].
            x_hat: Estimated data of shape [input_dim, dim_x+optionalDim].
            z_hat: Estimated data of shape [input_dim, dim_z+optionalDim].

        Returns:
            loss: loss1 + loss2
            loss_1: MSE(x,x_hat)
            loss_2: MSE(dTdx*f(x),D*z+F*h(x))
        """
        # Init mean squared error
        mse = nn.MSELoss()

        # Reconstruction loss MSE(x,x_hat)
        loss_1 = self.params['recon_lambda'] * mse(x, x_hat)

        # Compute gradients of T_u with respect to inputs
        dTdx = torch.autograd.functional.jacobian(
            self.encoder, x, create_graph=False, strict=False, vectorize=False)
        dTdx = dTdx[dTdx != 0].reshape((self.params['batch_size'], self.observer.dim_z, self.observer.dim_x))

        # lhs = dTdx * f(x)
        lhs = torch.zeros((self.observer.dim_z, self.params['batch_size']))
        for i in range(self.params['batch_size']):
            lhs[:, i] = torch.matmul(dTdx[i], self.observer.f(x.T).T[i]).T

        # rhs = D * z + F * h(x)
        D = self.observer.D.to(self.device)
        F = self.observer.F.to(self.device)
        h_x = self.observer.h(x.T).to(self.device)
        rhs = torch.matmul(D, z_hat.T) + torch.matmul(F, h_x)

        # PDE loss MSE(lhs, rhs)
        loss_2 = mse(lhs.to(self.device), rhs)

        return loss_1 + loss_2, loss_1, loss_2

    def loss_noise(self, x: torch.tensor, x_hat: torch.tensor, z_hat: torch.tensor) -> [
            torch.tensor, torch.tensor, torch.tensor]:
        """
        # WIP
        Loss function for noise experiment.

        Arguments:
            x: Input data of shape [input_dim, dim_x+optionalDim].
            x_hat: Estimated data of shape [input_dim, dim_x+optionalDim].
            z_hat: Estimated data of shape [input_dim, dim_z+optionalDim].

        Returns:
            loss: loss1 + loss2
            loss1: MSE(x,x_hat)
            loss2: MSE(dTdx*f(x),D*z+F*h(x))
            [optional loss]
            loss3: MSE(w_c, w_c_hat)
        """
        w_c = x[:, 0]
        w_c_hat = z_hat[:, 0]

        z = z_hat[:, 1:]

        mse = nn.MSELoss()

        loss1 = self.params['recon_lambda'] * mse(x, x_hat)

        # Compute gradients of T_u with respect to inputs
        dTdy = torch.autograd.functional.jacobian(self.encoder, x)
        dTdy = dTdy[dTdy != 0].reshape((self.params['batch_size'], self.observer.dim_z+1, self.observer.dim_x+1))
        dTdx = dTdy[:, 1:, 1:]

        lhs = torch.zeros((self.observer.dim_z, self.params['batch_size']))
        rhs = lhs.clone()

        for i in range(self.params['batch_size']):
            b, a = signal.bessel(3, w_c[i]*2*math.pi, 'low', analog=True, norm='phase')
            eigen = np.roots(a)

            # Place eigenvalue
            D = self.observer.tensorDFromEigen(eigen).to(self.device)

            lhs[:, i] = torch.matmul(dTdx[i], self.observer.f(x.T).T[i]).T
            rhs[:, i] = (torch.matmul(D, z[i].T).reshape(-1, 1) + torch.matmul(self.observer.F.to(self.device),
                         self.observer.h(x[i].reshape(-1, 1))).to(self.device)).squeeze()

        loss2 = mse(lhs.to(self.device), rhs)

        loss3 = mse(w_c, w_c_hat)

        loss = loss1 + loss2 + loss3

        return loss, loss1, loss2, loss3

    def forward(self, x: torch.tensor) -> [torch.tensor, torch.tensor]:
        """
        Forward function for autoencoder.
        z = encoder(x)
        x_hat = decoder(z)

        Arguments:
            x: Input data of shape [input_dim, dim_x+optionalDim].

        Returns:
            z: Latent space data of shape [input_dim, dim_z+optionalDim].
            x_hat: Estimated input data of shape [input_dim, dim_x+optionalDim].
        """
        # Enocde input space
        z = self.encoder(x)

        # Decode latent space
        x_hat = self.decoder(z)

        return z, x_hat
