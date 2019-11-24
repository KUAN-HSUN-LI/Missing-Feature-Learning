import pandas as pd
import torch


def f1_loss(predict, target):
    predict = torch.sigmoid(predict)
    predict = predict * (1-target) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return 1 - f1.mean()


def SubmitGenerator(prediction, sampleFile, filename='result/prediction.csv'):
    """
    Args:
        prediction (numpy array)
        sampleFile (str)
        filename (str)
    """
    sample = pd.read_csv(sampleFile)
    submit = dict()
    submit['Id'] = list(sample.Id.values)
    prediction_class = [int(p) for p in prediction]
    submit['Class'] = list(prediction_class)
    df = pd.DataFrame.from_dict(submit)
    df.to_csv(filename, index=False)
