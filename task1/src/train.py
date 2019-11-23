from loss import f1_loss
from trainer import Trainer
from preprocessor import preprocess_samples
from feature_dataset import FeatureDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from model.simpleNet import simpleNet
import torch

dataset = pd.read_csv("../data/train.csv")
dataset.drop("Id", axis=1, inplace=True)
dataset.drop("F1", axis=1, inplace=True)
train_set, valid_set = train_test_split(dataset, test_size=0.1, random_state=58)
train = preprocess_samples(train_set, missing=["F1"])
valid = preprocess_samples(valid_set, missing=["F1"])
trainData = FeatureDataset(train)
validData = FeatureDataset(valid)


model = simpleNet()
opt = torch.optim.Adam(model.parameters(), lr=2e-4)
criteria = f1_loss
max_epoch = 50
trainer = Trainer(trainData, validData, model, criteria, opt, "simpleNet1")

for epoch in range(max_epoch):
    print('Epoch: {}'.format(epoch))
    trainer.run_epoch(epoch, True)
    trainer.run_epoch(epoch, False)
    trainer.save(epoch)
