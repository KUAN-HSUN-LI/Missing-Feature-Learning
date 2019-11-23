from collections import Counter
import pandas as pd
import numpy as np


def counter(dataset):
    """
    """
    keys = Counter(dataset).keys()
    values = Counter(dataset).values()
    key_df = pd.DataFrame(data=keys, columns=["key"])
    value_df = pd.DataFrame(data=values, columns=["quntity"])
    percent_df = pd.DataFrame([i / len(data) * 100.0 for i in values], columns=["percentage"])

    df = pd.concat([key_df, value_df, percent_df], axis=1)

    print(df.round(3))


def get_outlier(data):
    """
    Args: 
        data (list or numpy): one feature of dataset
    Returns:
        list of outliers' index
    """
    tmp = np.percentile(data, (25, 50, 75), interpolation='midpoint')
    Q1 = tmp[0]
    Q2 = tmp[1]
    Q3 = tmp[2]
    QD = (Q3 - Q1) / 2
    max_value = Q3 + QD * 3
    min_value = Q1 - QD * 3
    outlier_idx = []

    for idx, d in enumerate(data):
        if d > max_value or d < min_value:
            outlier_idx.append(idx)
    return outlier_idx
