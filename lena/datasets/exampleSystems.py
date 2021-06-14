from lena.observer.lueneberger import LuenebergerObserver
import numpy as np
import torch


def getAutonomousSystem():
    # Define plant dynamics
    def f(x): return torch.cat((torch.reshape(torch.pow(x[1, :], 3), (1, -1)), torch.reshape(-x[0, :], (1, -1))), 0)
    def h(x): return torch.reshape(x[0, :], (1, -1))
    def g(x): return torch.zeros(x.shape[0], x.shape[1])
    def u(t): return 0

    # System dimension
    dim_x = 2
    dim_y = 1

    return f, h, g, dim_x, dim_y


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

    return f, h, g, u, dim_x, dim_y


def createDefaultObserver(params):
    if params['name'] == 'autonomous':
        f, h, g, u, dim_x, dim_y = getAutonomousSystem()
    elif params['name'] == 'van_der_pohl':
        f, h, g, u, dim_x, dim_y = getVanDerPohlSystem()

    # Initiate observer with system dimensions
    if params['experiment'] == 'autonomous':
        observer = LuenebergerObserver(dim_x, dim_y)
    elif params['experiment'] == 'noise':
        observer = LuenebergerObserver(dim_x, dim_y, 1)

    observer.f = f
    observer.h = h
    observer.g = g
    observer.u = u

    # Set system dynamics
    observer.F = torch.Tensor([[1.0], [1.0], [1.0]])

    return observer
