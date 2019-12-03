import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace as pdb


class simpleNet(nn.Module):
    def __init__(self, hidden_size):
        super(simpleNet, self).__init__()
        self.embedding = nn.Linear(11, hidden_size)

        self.out1 = nn.Linear(hidden_size, 3)
        self.out2 = nn.Linear(hidden_size, 2)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.embedding(x))
        # x = self.dropout(x)

        missing = self.out1(x)
        y = torch.sigmoid(self.out2(x))

        return missing, y

    def predict(self, x):
        x = F.relu(self.embedding(x))
        y = torch.sigmoid(self.out2(x))

        return y
