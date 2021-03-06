from torch.utils.data import Dataset
import torch


class FeatureDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):
        batch_feature = []
        batch_missing = []
        batch_label = []
        for data in datas:
            batch_feature.append(data['Features'])
            if 'Label' in data:
                batch_missing.append(data['Missing'])
                batch_label.append(data['Label'])

        return torch.FloatTensor(batch_feature), torch.FloatTensor(batch_missing), torch.FloatTensor(batch_label)
