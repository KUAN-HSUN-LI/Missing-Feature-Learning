import torch
from ipdb import set_trace as pdb


class F1:
    def __init__(self):
        self.threshold = 0.5
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0
        self.n_total = 0
        self.name = 'F1'

    def reset(self):
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0
        self.n_total = 0

    def update(self, predicts, groundTruth):
        self.find_max(predicts)
        self.n_precision += torch.sum(predicts).data.item()
        self.n_recall += torch.sum(groundTruth).data.item()
        self.n_corrects += torch.sum(groundTruth.cpu() * predicts).data.item()
        self.n_total += groundTruth.shape[0]

    def find_max(self, predicts):
        for predict in predicts:
            max_pos = torch.argmax(predict)
            for idx, num in enumerate(predict):
                if idx == max_pos:
                    predict[idx] = 1
                else:
                    predict[idx] = 0

    def get_score(self):
        recall = self.n_corrects / self.n_recall
        precision = self.n_corrects / (self.n_precision + 1e-20)
        return 2 * (recall * precision) / (recall + precision + 1e-20)

    def print_score(self):
        score = self.get_score()
        return '{:.5f}'.format(score)

    def get_accuracy(self):
        acc = float(self.n_corrects) / self.n_total
        return '{:.5f}'.format(acc)
