import pandas as pd
import statistics
import os


def SubmitGenerator(prediction, sampleFile, filename='result/prediction.csv'):
    """
    Args:
        prediction (numpy array)
        sampleFile (str)
        filename (str)
    """
    if not os.path.exists('result'):
        os.makedirs('result')
    sample = pd.read_csv(sampleFile)
    submit = dict()
    submit['Id'] = list(sample.Id.values)
    prediction_class = [int(p) for p in prediction]
    submit['Class'] = list(prediction_class)
    df = pd.DataFrame.from_dict(submit)
    df.to_csv(filename, index=False)


def get_outlier(data):
    """
    Args:
        data (list or numpy): one feature of dataset
    Returns:
        list of outliers' index
    """
    mean = sum(data) / len(data)
    std = statistics.stdev(data)
    max_value = mean + 3 * std
    min_value = mean - 3 * std
    outlier_idx = []

    for idx, d in enumerate(data):
        if d > max_value or d < min_value:
            outlier_idx.append(idx)
    return outlier_idx
