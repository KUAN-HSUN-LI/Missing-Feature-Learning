import pandas as pd


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
