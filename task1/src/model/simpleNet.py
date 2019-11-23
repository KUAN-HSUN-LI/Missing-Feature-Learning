import torch.nn as nn
import torch


class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()

        self.in1 = nn.Linear(9, 512)
        self.out = nn.Linear(512, 12)

    def forward(self, x):
        x = self.in1(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x
