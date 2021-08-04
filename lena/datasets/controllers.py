import numpy as np
import torch 
import scipy.signal as signal

# Reshape any vector of (length,) object to (1, length) (single point of
# certain dimension)
def reshape_pt1(x, verbose=False):
    if np.isscalar(x) or np.array(x).ndim == 0:
        x = np.reshape(x, (1, 1))
    else:
        x = np.array(x)
    if len(x.shape) == 1:
        x = np.reshape(x, (1, x.shape[0]))
    if verbose:
        print(x.shape)
    return x


# Reshape any point of type (1, length) to (length,)
def reshape_pt1_tonormal(x, verbose=False):
    if np.isscalar(x) or np.array(x).ndim == 0:
        x = np.reshape(x, (1,))
    elif len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0],))
    elif x.shape[0] == 1:
        x = np.reshape(x, (x.shape[1],))
    if verbose:
        print(x.shape)
    return x

# Reshape any vector of (length,) object to (length, 1) (possibly several
# points but of dimension 1)
def reshape_dim1(x, verbose=False):
    if np.isscalar(x) or np.array(x).ndim == 0:
        x = np.reshape(x, (1, 1))
    else:
        x = np.array(x)
    if len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0], 1))
    if verbose:
        print(x.shape)
    return x


# Reshape any vector of type (length, 1) to (length,)
def reshape_dim1_tonormal(x, verbose=False):
    if np.isscalar(x) or np.array(x).ndim == 0:
        x = np.reshape(x, (1,))
    elif len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0],))
    elif x.shape[1] == 1:
        x = np.reshape(x, (x.shape[0],))
    if verbose:
        print(x.shape)
    return x


# Possible controllers

# Sinusoidal control law, imposing initial value
def sin_controller(t):
    t = np.asscalar(t.numpy())
    t0 = 0
    init_control = 0
    gamma=0.4
    omega=1.2
    if np.isscalar(t):
        if t == t0:
            u = 0.
        else:
            u = gamma * np.cos(omega * t)
    else:
        u = reshape_pt1(np.concatenate((reshape_dim1(np.zeros(len(t))),
                                        reshape_dim1(
                                            gamma * np.cos(omega * t))),
                                       axis=1))
        if t[0] == t0:
            u[0] = reshape_pt1(init_control)
    return torch.tensor(u)


# Homemade chirp control law, imposing initial value
def chirp_controller(t, kwargs, t0, init_control):
    gamma = kwargs.get('gamma')
    f0 = kwargs.get('f0')
    f1 = kwargs.get('f1')
    t1 = kwargs.get('t1')
    nb_cycles = int(np.floor(np.min(t) / t1))
    t = t - nb_cycles * t1
    if np.isscalar(t):
        if t == t0:
            u = reshape_pt1(init_control)
        else:
            u = reshape_pt1(
                [[0, signal.chirp(t, f0=f0, f1=f1, t1=t1, method='linear')]])
    else:
        u = reshape_pt1(np.concatenate((reshape_dim1(np.zeros(len(t))),
                                        reshape_dim1(
                                            signal.chirp(t, f0=f0, f1=f1, t1=t1,
                                                         method='linear'))),
                                       axis=1))
        if t[0] == t0:
            u[0] = reshape_pt1(init_control)
    return torch.tensor(gamma * u)


# Linear chirp control law, imposing initial value
def lin_chirp_controller(t):
    t = np.asscalar(t.numpy())
    t0 = 0
    init_control = 0
    a = 0.001
    b = 9.99e-05
    if np.isscalar(t):
        if t == t0:
            u = 0.
        else:
            u = np.sin(2 * np.pi * t * (a + b * t))
    else:
        u = reshape_pt1(np.concatenate((reshape_dim1(np.zeros(len(t))),
                                        reshape_dim1(np.sin(2 * np.pi * t * (
                                                a + b * t)))), axis=1))
        if t[0] == t0:
            u[0] = reshape_pt1(init_control)
    return torch.tensor(u)
