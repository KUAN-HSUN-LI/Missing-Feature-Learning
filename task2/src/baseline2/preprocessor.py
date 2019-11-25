from tqdm import tqdm


def preprocess_samples(dataset, missing=None):
    """ Worker function.

    Args:
        dataset (list of dict)
    Returns:
        list of processed dict.
    """
    processed = []
    for sample in tqdm(dataset.iterrows(), total=len(dataset), ascii=True):
        processed.append(preprocess_sample(sample[1], missing))

    return processed


def preprocess_sample(data, missing):
    """
    Args:
        data (dict)
    Returns:
        dict
    """
    features = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14']
    if missing:
        for m in missing:
            features.remove(m)
    processed = dict()
    processed['Features'] = [data[feature] for feature in features]
    if 'Class' in data:
        processed['Missing'] = [data[m] for m in missing]
        processed['Label'] = [1, 0] if data['Class'] == 0 else [0, 1]

    return processed
