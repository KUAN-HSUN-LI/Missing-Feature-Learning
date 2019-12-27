import torch.nn as nn
import torch


class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(SimpleNet, self).__init__()

        self.input = nn.Linear(input_size, hidden_size)
        self.BN = nn.BatchNorm1d(hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.BN1 = nn.BatchNorm1d(hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.BN2 = nn.BatchNorm1d(hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.BN3 = nn.BatchNorm1d(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out1 = torch.relu(self.input(x))
        x = self.BN(out1)
        x = self.dropout(x)
        out2 = torch.relu(self.l1(x))
        x = self.BN1(out2)
        x = self.dropout(x)
        out3 = torch.relu(self.l2(x))
        x = self.BN2(out3)
        x = self.dropout(x)
        out4 = torch.relu(self.l3(x))
        x = self.BN3(out4)
        x = self.dropout(x)
        out = self.out(x)
        return out, (out1, out2, out3, out4)
