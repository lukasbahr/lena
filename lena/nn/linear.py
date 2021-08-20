import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, observer, params, device):
        super().__init__()

        # Set x and z dimension
        dim_x = observer.dim_x 
        dim_z = observer.dim_z 
        dim_in = dim_z + observer.optionalDim

        self.fc1 = nn.Linear(dim_in, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 40)
        self.fc5 = nn.Linear(40, 40)
        self.fc6 = nn.Linear(40, dim_x)
        self.tanh = nn.Tanh()

        self.params = params
        self.device = device

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.tanh(self.fc5(x))
        x = self.fc6(x)
        return x

    def loss(self, x, x_hat):
        # Init mean squared error
        mse = nn.MSELoss()

        # Reconstruction loss MSE(x,x_hat)
        loss_1 =  mse(x, x_hat)

        return loss_1

    def predict(self, params, observer, w_c, var):
        tsim = params['t_sim']
        dt = params['dt']

        observer.set_D(w_c)

        # Get measurements fom x forward in time
        init_cond = torch.Tensor(params['init_condition'])
        w_0_truth = torch.cat((init_cond, torch.tensor([[0., 0., 0.]]).T),0)

        tq_, w_truth = observer.simulateSystem(w_0_truth, tsim, dt)

        # Solve z_dot with measurement y
        y = torch.cat((tq_.unsqueeze(1), observer.h(
            w_truth[:, :observer.dim_x, 0].T).T), dim=1)
        tq_pred, w_observer = observer.simulateLueneberger(y, tsim, dt)
        w_observer = w_observer[:, :, 0]

        noise = torch.normal(0, var, size=(w_observer.shape[0], 3))

        w_observer = w_observer.add(noise)

        w_c = w_c.repeat(w_observer.shape[0]).unsqueeze(1)
        data = torch.cat((w_observer, w_c), 1)

        # Predict x_hat with T_star(z)
        with torch.no_grad():
            x_hat = self.forward(data)

        return tq_pred, x_hat, data, w_truth
