from tqdm import tqdm
import torch


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


def preprocess_samples(dataset, feature_for_training=None, feature_for_prediction=None):
    """ Worker function.
    Args:
        dataset (list of dict)
    Returns:
        list of processed dict.
    """
    processed = []
    for sample in tqdm(dataset.iterrows(), total=len(dataset)):
        processed.append(preprocess_sample(sample[1], feature_for_training, feature_for_prediction))

    return processed


def preprocess_sample(data, feature_for_training=None, feature_for_prediction=None):
    """
    Args:
        data (dict)
    Returns:
        dict
    """
    processed = {}
    processed['Feature'] = [data[feature] for feature in feature_for_training]
    if feature_for_prediction is not None:
        processed['Label_feature'] = [data[feature] for feature in feature_for_prediction]
    if 'Class' in data:
        processed['Label'] = label_to_idx(data['Class'])

    return processed
