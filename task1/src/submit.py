

def SubmitGenerator(prediction, sampleFile, filename='prediction.csv'):
    """
    Args:
        prediction (numpy array)
        sampleFile (str)
        filename (str)
    """
    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'W': 9, 'X': 10, 'Y': 11}

    key_list = list(label_dict.keys())

    sample = pd.read_csv(sampleFile)
    submit = {}
    submit['Id'] = list(sample.Id.values)
    prediction_class = [key_list[p] for p in prediction]
    submit['Class'] = list(prediction_class)
    df = pd.DataFrame.from_dict(submit)
    df.to_csv(filename, index=False)
