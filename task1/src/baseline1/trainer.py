import os
import json
from tqdm import tqdm
from tqdm import trange
import torch
from metrics import accuracy
from torch.utils.data import DataLoader


class Trainer():
    def __init__(self, device, trainData, validData, model, criteria, opt, batch_size, model_dir="model"):
        self.device = device
        self.history = {'train': [], 'valid': []}
        self.trainData = trainData
        self.validData = validData
        self.model = model.to(self.device)
        self.criteria = criteria
        self.opt = opt
        self.batch_size = batch_size
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_val = 0.0

    def run_epoch(self, epoch, training):
        self.model.train(training)

        if training:
            description = 'Train'
            dataset = self.trainData
            shuffle = True
        else:
            description = 'Valid'
            dataset = self.validData
            shuffle = False
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=shuffle,
                                collate_fn=dataset.collate_fn,
                                num_workers=4)

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)

        loss = 0
        acc = accuracy()

        for i, (x, _, y) in trange:
            o_labels, batch_loss = self.run_iter(x, y)
            if training:
                self.opt.zero_grad()
                batch_loss.backward()
                self.opt.step()

            loss += batch_loss.item()
            acc.update(o_labels.cpu(), y)

            trange.set_postfix(
                loss=loss / (i + 1), acc=acc.print_score())
        if training:
            self.history['train'].append({'acc': acc.get_score(), 'loss': loss / len(trange)})
            self.save_hist()
        else:
            self.history['valid'].append({'acc': acc.get_score(), 'loss': loss / len(trange)})
            self.save_hist()
            if acc.get_score() > self.best_val:
                self.best_val = acc.get_score()
                self.save_best(epoch)

    def run_iter(self, x, y):
        features = x.to(self.device)
        labels = y.to(self.device)
        o_labels = self.model(features)
        l_loss = self.criteria(o_labels, labels)
        return o_labels, l_loss

    def save_best(self, epoch):
        torch.save(self.model.state_dict(), self.model_dir + '/model.pkl.'+str(epoch))

    def save_hist(self):
        with open(self.model_dir + '/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
