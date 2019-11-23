from tqdm import tqdm_notebook as tqdm


def label_to_onehot(labels):
    """ Convert label to onehot .
    Args:
        labels (string): data's labels.
    Return:
        outputs (onehot list): data's onehot label.
    """
    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'W': 9, 'X': 10, 'Y': 11}
    onehot = [0] * len(label_dict)
    for l in labels.split():
        onehot[label_dict[l]] = 1
    return onehot


def label_to_idx(labels):
    """
    Args:
        labels (string): data's labels.
    Return:
        outputs (int): index of data's label 
    """
    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'W': 9, 'X': 10, 'Y': 11}
    return label_dict[labels]


def preprocess_samples(dataset):
    """ Worker function.

    Args:
        dataset (list of dict)
    Returns:
        list of processed dict.
    """
    processed = []
    for sample in tqdm(dataset.iterrows(), total=len(dataset)):
        processed.append(preprocess_sample(sample[1]))

    return processed


def preprocess_sample(data):
    """
    Args:
        data (dict)
    Returns:
        dict
    """
    features = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
    processed = {}
    processed['Features'] = [data[feature] for feature in features]
    if 'Class' in data:
        processed['Label'] = label_to_onehot(data['Class'])

    return processed
