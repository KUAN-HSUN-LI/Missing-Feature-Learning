import torch


class F1():
    def __init__(self):
        self.threshold = 0.5
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0
        self.name = 'F1'

    def reset(self):
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0

    def update(self, predicts, groundTruth):
        self.find_max(predicts)
        self.n_precision += torch.sum(predicts).data.item()
        self.n_recall += torch.sum(groundTruth).data.item()
        self.n_corrects += torch.sum(groundTruth.cpu() * predicts).data.item()

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
