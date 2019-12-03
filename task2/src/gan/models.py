import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace as pdb


class Generator(nn.Module):
    def __init__(self, hidden_size):
        super(Generator, self).__init__()
        self.input = nn.Linear(11, hidden_size)
        self.output = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.output(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        self.input = nn.Linear(3, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleNet, self).__init__()

        self.in1 = nn.Linear(14, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.in1(x))
        x = self.out(x)
        return x
