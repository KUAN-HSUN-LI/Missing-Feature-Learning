import torch.nn as nn
import torch


class simpleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(simpleNet, self).__init__()

        self.in1 = nn.Linear(input_size, 512)
        self.BN = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.in1(x)
        x = torch.relu(x)
        x = self.BN(x)
        x = self.out(x)
        return x
