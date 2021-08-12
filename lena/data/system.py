from scipy import signal
from smt.sampling_methods import LHS
from torchdiffeq import odeint
from math import pi
import numpy as np
import torch

class System():
    def __init__(self):
        self.u = self.null_controller

    def set_controller(self, controller):
        if controller == 'null_controller':
            self.u = self.null_controller
        elif controller == 'sin_controller':
            self.u = self.sin_controller
        elif controller == 'lin_chirp_controller':
            self.u = self.lin_chirp_controller

    def mesh(self, limits: tuple, num_samples: int, method='lhs'):
        # Sample either a uniformly grid or use latin hypercube sampling
        if limits[1] < limits[0]:
            raise ValueError('limits[0] must be strictly smaller than limits[0]')

        if method == 'uniform':
            grid_step = (limits[1]-limits[0]) / np.sqrt(num_samples)
            axes = np.arange(limits[0], limits[1], grid_step)
            mesh = np.array(np.meshgrid(axes, axes)).T.reshape(-1, 2)

        elif method == 'lhs':
            limits = np.array([limits, limits])
            sampling = LHS(xlimits=limits)
            mesh = sampling(num_samples)

        return torch.from_numpy(mesh)

    def mesh_noise(self, limits: tuple, limits_wc: tuple, num_samples: int):

        if limits[1] < limits[0]:
            raise ValueError('limits[0] must be strictly smaller than limits[0]')
        if limits_wc[1] < limits_wc[0]:
            raise ValueError('limits_wc[0] must be strictly smaller than limits_wc[0]')

        limits = np.array([limits_wc, limits, limits])
        sampling = LHS(xlimits=limits)
        mesh = torch.Tensor(sampling(num_samples))

        return mesh 

    def simulate(self, x_0: torch.tensor, tsim: tuple, dt) -> [torch.tensor, torch.tensor]:
        """
        Runs and outputs the results from 
        multiple simulations of an input-affine nonlinear system driving a 
        Luenberger observer target system.

        Arguments:
            y_0: Initial value for system simulation.
            tsim: Tuple of (Start, End) time of simulation.
            dt: Step width of tsim.

        Returns:
            tq: Array of timesteps of tsim.
            sol: Solver solution.
        """
        def dydt(t, x):
            x_dot = self.f(x) + self.g(x) * self.u(t)
            return x_dot

        # Output timestemps of solver
        tq = torch.arange(tsim[0], tsim[1], dt)

        # Solve
        sol = odeint(dydt, x_0, tq)

        return tq, sol

    def lin_chirp_controller(self, t, t_0=0, a=0.001, b=9.99e-05):
        if t == t_0:
            u = 0.
        else:
            u = torch.sin(2 * pi * t * (a + b * t))
        return u

    def sin_controller(self, t, t0=0, init_control=0, gamma=0.4, omega=1.2):
        if t == t0:
            u = 0.
        else:
            u = gamma * torch.cos(omega * t)
        return u

    def chirp_controller(self, t, t_0=0, f_0=6, f_1=1, t_1=10, gamma=1):
        t = t.numpy()
        nb_cycles = int(np.floor(np.min(t) / t_1))
        t = t - nb_cycles * t_1
        if t == t_0:
            u = 0.
        else:
            u = signal.chirp(t, f0=f_0, f1=f_1, t1=t_1, method='linear')
        return torch.tensor(gamma * u)

    def null_controller(self, t):
        return 0.


class ClassicRevDuffing(System):

    def __init__(self):
        self.dim_x = 2
        self.dim_y = 1

    def f(self, x):
        x_0 = torch.reshape(torch.pow(x[1, :], 3), (1, -1))
        x_1 = torch.reshape(-x[0, :], (1, -1))
        return torch.cat((x_0, x_1), 0)

    def h(self, x):
        return torch.reshape(x[0, :], (1, -1))

    def g(self, x):
        return torch.zeros(x.shape[0], x.shape[1])


class ClassicVanDerPohl(System):

    def __init__(self, eps=1):
        self.dim_x = 2
        self.dim_y = 1

        self.eps = eps

    def f(self, x):
        x_0 = torch.reshape(x[1, :], (1, -1))
        x_1 = torch.reshape(self.eps*(1-torch.pow(x[0, :], 2))*x[1, :]-x[0, :], (1, -1))
        return torch.cat((x_0, x_1))

    def h(self, x):
        return torch.reshape(x[0, :], (1, -1))

    def g(self, x):
        zeros = torch.reshape(torch.zeros_like(x[1, :]), (1, -1))
        ones = torch.reshape(torch.ones_like(x[0, :]), (1, -1))
        return torch.cat((zeros, ones))
