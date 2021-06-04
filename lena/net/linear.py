import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils




class Model(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, outputSize)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.fc4(x)
        return x
