from tqdm import tqdm


def label_to_onehot(labels):
    """ Convert label to onehot .
    Args:
        labels (string): data's labels.
    Return:
        outputs (onehot list): data's onehot label.
    """
    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}
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
    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}
    return label_dict[labels]


def preprocess_samples(dataset, label_for_training=None, label_for_predict=None):
    """ Worker function.
    Args:
        dataset (list of dict)
    Returns:
        list of processed dict.
    """
    processed = []
    for sample in tqdm(dataset.iterrows(), total=len(dataset)):
        processed.append(preprocess_sample(sample[1], label_for_training, label_for_predict))

    return processed


def preprocess_sample(data, label_for_training=None, label_for_predict=None):
    """
    Args:
        data (dict)
    Returns:
        dict
    """
    processed = {}
    if label_for_training is not None:
        processed['Features'] = [data[feature] for feature in label_for_training]
    if label_for_predict is not None:
        processed['Label_features'] = [data[feature] for feature in label_for_predict]
    if 'Class' in data:
        processed['Label'] = label_to_idx(data['Class'])
    return processed
