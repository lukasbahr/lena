import numpy as np
from scipy import linalg
from torchdiffeq import odeint
from scipy.interpolate import interp1d
from torchinterp1d import Interp1d
import torch


class LuenebergerObserver():
    def __init__(self, dim_x: int, dim_y: int, optionalDim=0):
        """
        Constructor for setting the dynamics of the Luenberger Observer. 
        Also constructs placeholder for D and F matrices.

        Arguments:
            dim_x -- dimension of states
            dim_y -- dimension of inputs
            optionalDim -- additional dimension for experiments
        Returns:
            None
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_y * (dim_x + 1)
        self.optionalDim = optionalDim

        self.F = torch.zeros((self.dim_z, 1))
        self.eigenD = torch.zeros((self.dim_z, 1))
        self.D = torch.zeros((self.dim_z, self.dim_z))

    def f(self, x): return 0
    def g(self, x): return 0
    def h(self, x): return 0
    def u(self, x): return 0
    def e(self, t): return 0

    def tensorDFromEigen(self, eigen: torch.tensor) -> torch.tensor:
        """
        Return matrix D as conjugate block matrix from eigenvectors 
        in form of conjugate complex numbers. 

        Arguments:
            eigen -- dimension of states

        Returns:
            D -- conjugate block matrix
        """
        self.eigenD = eigen
        eig_complex, eig_real = [x for x in eigen if x.imag != 0], [
            x for x in eigen if x.imag == 0]

        if(any(~np.isnan(eig_complex))):
            eig_complex = sorted(eig_complex)
            eigenCell = self.eigenCellFromEigen(eig_complex, eig_real)
            D = linalg.block_diag(*eigenCell[:])

            return torch.from_numpy(D).float()

    @staticmethod
    def eigenCellFromEigen(eig_complex: torch.tensor, eig_real: torch.tensor) -> []:
        """
        Generates a cell array containing 2X2 real
        matrices for each pair of complex conjugate eigenvalues passed in the
        arguments, and real scalar for each real eigenvalue.

        Arguments:
            eigen -- dimension of states

        Returns:
            array -- array of conjugate pairs of eigenvectors
        """
        eigenCell = []

        for i in range(0, len(eig_complex), 2):
            array = np.zeros(shape=(2, 2))
            array[0, 0] = eig_complex[i].real
            array[0, 1] = eig_complex[i].imag
            array[1, 0] = eig_complex[i+1].imag
            array[1, 1] = eig_complex[i+1].real
            eigenCell.append(array)

        for i in eig_real:
            array = np.zeros(shape=(1, 1))
            array[0, 0] = i.real
            eigenCell.append(array)

        return eigenCell

    def simulateLueneberger(self, y_0: torch.tensor, tsim: tuple, dt) -> [torch.tensor, torch.tensor]:
        """
        Runs and outputs the results from 
        multiple simulations of an input-affine nonlinear system driving a 
        Luenberger observer target system.

        Arguments:
            y_0 -- initial value
            tsim -- tuple of (start,end)
            dt -- step width

        Returns:
            tq -- array timesteps
            sol -- solver solution for x_dot and z_dot
        """

        def dydt(t, y):
            x = y[0:self.dim_x]
            z = y[self.dim_x:len(y)]
            x_dot = self.f(x) + self.g(x) * self.u(t)
            z_dot = torch.matmul(self.D, z)+self.F*self.h(x)+self.F*self.e(t)
            return torch.cat((x_dot, z_dot))

        # Output timestemps of solver
        tq = torch.arange(tsim[0], tsim[1], dt)

        # Solve
        sol = odeint(dydt, y_0, tq)

        return tq, sol

    def simulateZ(self, y, tsim: tuple, dt) -> [torch.tensor, torch.tensor]:
        """
        Runs and outputs the results from 
        multiple simulations of an input-affine nonlinear system driving a 
        Luenberger observer target system.

        Arguments:
            y_0 -- initial value
            tsim -- tuple of (start,end)
            dt -- step width

        Returns:
            tq -- array timesteps
            sol -- solver solution for x_dot and z_dot
        """

        def dydt(t, z):
            z_dot = torch.matmul(self.D, z)+self.F*self.measurement(t)
            return z_dot

        # Output timestemps of solver
        tq = torch.arange(tsim[0], tsim[1], dt)

        # Intepolation of y
        self.measurement = self.interpolate_func(y)

        # Zero initial value
        z_0 = torch.zeros((self.dim_z,1))

        # Solve
        sol = odeint(dydt, z_0, tq)

        return tq, sol

    # Vector x = (t_i, x(t_i)) of time steps t_i at which x is known is interpolated at given
    # time t, interpolating along each output dimension independently if there
    # are more than one. Returns a function interp(t) which interpolates x at times t
    def interpolate_func(self, x, method='linear') -> callable:
        """Takes a vector of times and values, returns a callable function which
        interpolates the given vector (along each output dimension independently).
        :param x: vector of (t_i, x(t_i)) to interpolate
        :type x: torch.tensor
        :returns: function interp(t, other args) which interpolates x at t
        :rtype:  Callable[[List[float]], np.ndarray]
        Author: Mona Buisson-Fenet 
        """
        if torch.is_tensor(x):  # not building computational graph!
            with torch.no_grad():
                if method != 'linear':
                    raise NotImplementedError(
                        'Only linear interpolator available in pytorch!')
                points, values = x[:, 0].contiguous(), x[:, 1:].contiguous()
                interp_list = [Interp1d() for i in range(values.shape[1])]
                def interp(t):
                    if len(t.shape) == 0:
                        t = t.reshape(1, ).contiguous()
                    else:
                        t = t.contiguous()
                    if len(x) == 1:
                        # If only one value of x available, assume constant
                        interpolate_x = x[0, 1:].repeat(len(t), 1)
                    else:
                        res = [interp_list[i](points, values[:, i], t) for i
                            in range(values.shape[1])]
                        interpolate_x = torch.squeeze(torch.stack(res), dim=1).t()
                    return interpolate_x
        else:
            points, values = x[:, 0], x[:, 1:]
            interp_list = [interp1d(x=points, y=values[:, i], kind=method)
                        for i in range(values.shape[1])]
            def interp(t):
                if np.isscalar(t):
                    t = np.array([t])
                if len(x) == 1:
                    # If only one value of x available, assume constant
                    interpolate_x = np.tile(reshape_pt1(x[0, 1:]), (len(t), 1))
                else:
                    interpolate_x = np.array([f(t) for f in interp_list]).T
                return interpolate_x
        return interp
    