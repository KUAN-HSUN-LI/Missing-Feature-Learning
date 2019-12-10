import torch.nn as nn
import torch


class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(SimpleNet, self).__init__()

        self.input = nn.Linear(input_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = self.dropout(x)
        x = torch.relu(self.l1(x))
        x = self.dropout(x)
        x = torch.relu(self.l2(x))
        x = self.dropout(x)
        x = torch.relu(self.l3(x))
        x = self.dropout(x)
        x = self.out(x)
        return x
