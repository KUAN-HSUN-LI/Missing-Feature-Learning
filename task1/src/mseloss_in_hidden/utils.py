import pandas as pd
from collections import Counter


def SubmitGenerator(prediction, sampleFile, filename='result/prediction.csv'):
    """
    Args:
        prediction (numpy array)
        sampleFile (str)
        filename (str)
    """
    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}
    key_list = list(label_dict.keys())

    sample = pd.read_csv(sampleFile)
    submit = {}
    submit['Id'] = list(sample.Id.values)
    prediction_class = [key_list[p] for p in prediction]
    submit['Class'] = list(prediction_class)
    df = pd.DataFrame.from_dict(submit)
    df.to_csv(filename, index=False)


def counter(dataset):
    """Account for the classes in the data
    Args:
        dataset (list, numpy)
    """
    keys = Counter(dataset).keys()
    values = Counter(dataset).values()
    key_df = pd.DataFrame(data=keys, columns=["key"])
    value_df = pd.DataFrame(data=values, columns=["quntity"])
    percent_df = pd.DataFrame([i / len(dataset) * 100.0 for i in values], columns=["percentage"])

    df = pd.concat([key_df, value_df, percent_df], axis=1)

    print(df.round(3))
