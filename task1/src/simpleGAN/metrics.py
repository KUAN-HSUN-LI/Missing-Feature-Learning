import torch


class accuracy():
    def __init__(self):
        self.ndata = 0
        self.ndata_correct = 0

    def update(self, predictions, trueIdxs):
        self.ndata += len(trueIdxs)
        for prediction, trueIdx in zip(predictions, trueIdxs):
            if torch.argmax(prediction) == trueIdx:
                self.ndata_correct += 1

    def get_score(self):
        return self.ndata_correct / self.ndata

    def print_score(self):
        score = self.get_score()
        return '{:.5f}'.format(score)
