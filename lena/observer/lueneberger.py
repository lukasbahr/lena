import numpy as np
from scipy import linalg
from torchdiffeq import odeint
from torchinterp1d import Interp1d
import torch

class LuenebergerObserver():
    """
    Class for Lueneberger Observer [https://en.wikipedia.org/wiki/State_observer].
    """

    def __init__(self, dim_x: int, dim_y: int, optionalDim=0):
        """
        Constructor for setting the dimensions of the Luenberger Observer. 
        Also constructs placeholder for D and F matrices.

        Arguments:
            dim_x: Dimension of states.
            dim_y: Dimension of inputs.
            optionalDim: Additional dimensions for experiments.
        """
        # Set observer dimensions
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_y * (dim_x + 1)
        self.optionalDim = optionalDim

        # Set observer matrices D and F
        self.F = torch.zeros((self.dim_z, 1))
        self.D = torch.zeros((self.dim_z, self.dim_z))

        # Eigenvalues of D as conjugate pairs
        self.eigenD = torch.zeros((self.dim_z, 1))

    # Dynamical function
    def f(self, x): return 0
    # Measurement vector
    def g(self, x): return 0
    # Control vector on input
    def h(self, x): return 0
    # Input on dynamical system
    def u(self, x): return 0
    # Noise on observation
    def e(self, t): return 0

    def tensorDFromEigen(self, eigen: torch.tensor) -> torch.tensor:
        """
        Return matrix D as conjugate block matrix from eigenvectors 
        in form of conjugate complex numbers. 

        Arguments:
            eigen: Dimension of states.

        Returns:
            D: Conjugate block matrix
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
            eigen: Dimension of states.

        Returns:
            array: Array of conjugate pairs of eigenvectors.
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

    def simulateSystem(self, y_0: torch.tensor, tsim: tuple, dt) -> [torch.tensor, torch.tensor]:
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

    def simulateLueneberger(self, y, tsim: tuple, dt, model=None, u_0=0) -> [torch.tensor, torch.tensor]:
        """
        Runs and outputs the results from Luenberger observer system.

        Arguments:
            y_0: Initial value for observer simulation.
            tsim: Tuple of (Start, End) time of simulation.
            dt: Step width of tsim.

        Returns:
            tq: Array of timesteps of tsim.
            sol: Solver solution.
        """
        def dydt(t, z):
            if self.model is None:
                z_dot = torch.matmul(self.D, z)+self.F*self.measurement(t)
            else:
                z_dot = torch.matmul(self.D, z)+self.F*self.measurement(t)+torch.mul(self.phi(z),self.u(t)-self.u_0(t))
            return z_dot

        # Output timestemps of solver
        tq = torch.arange(tsim[0], tsim[1], dt)

        # Phi
        self.model = model
        if self.model is not None:
            def phi(z): 
                dTdy = torch.autograd.functional.jacobian(
                    model.encoder, self.model.decoder(z.T), create_graph=False, strict=False, vectorize=False)

                dTdx = torch.zeros((self.dim_z, self.dim_x))
                for j in range(dTdy.shape[1]):
                    dTdx[j, :] = dTdy[0, j, 0, :]

                affine = self.g(self.model.decoder(z.T).T)
                return torch.matmul(dTdx, affine)

            self.phi = phi
            self.u_0 = u_0

        # 1D interpolation of y
        self.measurement = self.interpolateFunc(y)

        # Zero initial value
        z_0 = torch.zeros((self.dim_z,1))

        # Solve
        sol = odeint(dydt, z_0, tq)

        return tq, sol

    @staticmethod
    def interpolateFunc(x, method='linear') -> callable:
        """Takes a vector of times and values, returns a callable function which
        interpolates the given vector (along each output dimension independently).

        Author: Mona Buisson-Fenet 

        Arguments:
            x: Vector of (t_i, x(t_i)) to interpolate.
            method: Interpolation method.
        
        Returns:
            Callable[[List[float]], np.ndarray] function.
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
        return interp
    