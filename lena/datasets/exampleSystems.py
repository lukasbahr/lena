from lnos.observer.lueneberger import LuenebergerObserver
from scipy import signal
import numpy as np
import math
import torch


def getAutonomousSystem():
    # Define plant dynamics
    def f(x): return torch.cat((torch.reshape(torch.pow(x[1, :], 3), (1, -1)), torch.reshape(-x[0, :], (1, -1))), 0)
    def h(x): return torch.reshape(x[0, :], (1, -1))
    def g(x): return torch.zeros(x.shape[0], x.shape[1])
    def u(x): return 0

    # System dimension
    dim_x = 2
    dim_y = 1

    # Eigenvalues for D
    b, a = signal.bessel(3, 2*math.pi, 'low', analog=True, norm='phase')
    eigen = np.roots(a)

    return f, h, g, u, dim_x, dim_y, eigen


def getVanDerPohlSystem():
    # Define plant dynamics
    eps = 1
    def f(x): return torch.cat((torch.reshape(x[1, :], (1, -1)),
                                torch.reshape(eps*(1-torch.pow(x[0, :], 2))*x[1, :]-x[0, :], (1, -1))))
    def h(x): return torch.reshape(x[0, :], (1, -1))
    def g(x): return torch.cat((torch.reshape(torch.zeros_like(
        x[1, :]), (1, -1)), torch.reshape(torch.ones_like(x[0, :]), (1, -1))))

    def u(t): return 10e-3 + 9.99 * 10e-5*t

    # System dimension
    dim_x = 2
    dim_y = 1

    # Eigenvalues for D
    b, a = signal.bessel(3, 2*math.pi, 'low', analog=True, norm='phase')
    eigen = np.roots(a)

    return f, h, g, u, dim_x, dim_y, eigen


def createDefaultObserver(options):
    if options['system'] == 'autonomous':
        f, h, g, u, dim_x, dim_y, eigen = getAutonomousSystem()
    elif options['system'] == 'van_der_pohl':
        f, h, g, u, dim_x, dim_y, eigen = getVanDerPohlSystem()
    else:
        print("Can't find system {}. Available options ['autonomous', 'van_der_pohl']".format(options['system']))

    # Initiate observer with system dimensions
    observer = LuenebergerObserver(dim_x, dim_y, f, g, h, u)

    # Set system dynamics
    observer.D = observer.tensorDFromEigen(eigen)
    observer.F = torch.Tensor([[1.0], [1.0], [1.0]])

    return observer
