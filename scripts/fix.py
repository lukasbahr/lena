import torch

# x[3,2]
def f(x):
    x_00 = torch.reshape(torch.pow(x[0][0], 2), (1, -1))
    x_01 = torch.reshape(torch.pow(x[1][0], 2), (1, -1))
    x_02 = torch.reshape(torch.pow(x[2][0], 2), (1, -1))
    x_10 = torch.reshape(torch.pow(x[0][1], 1), (1, -1))
    x_11 = torch.reshape(torch.pow(x[1][1], 1), (1, -1))
    x_12 = torch.reshape(torch.pow(x[2][1], 1), (1, -1))
    x_ = torch.cat((x_00,x_01,x_02),0)
    x__ = torch.cat((x_10,x_11,x_12),0)
    return torch.cat((x_,x__), 1)

if __name__ == "__main__":
    # x = torch.tensor([[3.,2.],[3.,2.],[3.,2.],[3.,2.]])
    x = torch.tensor([[2.,3.],[2.,3.],[2.,3.]])
    print(f(x))

    dTdh = torch.autograd.functional.jacobian(f, x)
    print(dTdh)
    
    dTdy = torch.zeros((x.shape[1], x.shape[0], x.shape[1]))
    
    # # [4, 2, 4, 2] --> [2, 4, 2]
    for i in range(dTdy.shape[0]):
        for j in range(dTdy.shape[1]):
            dTdy[i, j, :] = dTdh[i, j, i, :]
    
    print(dTdy)