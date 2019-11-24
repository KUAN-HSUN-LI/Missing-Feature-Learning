import torch
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
        self.n_corrects += torch.sum(groundTruth.argmax(dim=1) == predicts.argmax(dim=1)).item()
        self.n_total += groundTruth.shape[0]

    def get_score(self):
        acc = float(self.n_corrects) / self.n_total
        return '{:.5f}'.format(acc)
