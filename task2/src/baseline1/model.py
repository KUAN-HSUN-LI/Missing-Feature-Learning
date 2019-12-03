import torch
import torch.nn as nn


class simpleNet(nn.Module):
    def __init__(self, hidden_size):
        super(simpleNet, self).__init__()

        self.in1 = nn.Linear(11, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.in1(x))
        x = self.out(x)
        # x = torch.sigmoid(x)
        return x
