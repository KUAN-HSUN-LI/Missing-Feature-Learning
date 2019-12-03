import torch
import torch.nn as nn
from ipdb import set_trace as pdb


class simpleNet(nn.Module):
    def __init__(self, hidden_size):
        super(simpleNet, self).__init__()

        self.in1 = nn.Linear(11, hidden_size)
        self.out1 = nn.Linear(hidden_size, 3)

        self.in2 = nn.Linear(14, hidden_size)
        self.out2 = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        missing = torch.relu(self.in1(x))
        # missing = self.dropout(missing)
        missing = self.out1(missing)
        y = torch.cat((x, missing), dim=1)

        y = torch.relu(self.in2(y))
        # y = self.dropout(y)
        y = self.out2(y)
        # y = torch.sigmoid(y)
        return missing, y
