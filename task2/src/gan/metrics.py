import torch
import torch.nn.functional as F
from ipdb import set_trace as pdb


class Accuracy:
    def __init__(self):
        self.n_corrects = 0
        self.n_total = 0
        self.name = 'Accuracy'

    def reset(self):
        self.n_corrects = 0
        self.n_total = 0

    def update(self, predicts, groundTruth):
        predicts = F.sigmoid(predicts) > 0.5
        groundTruth = groundTruth > 0.5
        self.n_corrects += torch.sum(predicts == groundTruth).item()
        self.n_total += groundTruth.shape[0]

    def get_score(self):
        acc = float(self.n_corrects) / self.n_total
        return acc

    def print_score(self):
        return '{:.5f}'.format(self.get_score())
