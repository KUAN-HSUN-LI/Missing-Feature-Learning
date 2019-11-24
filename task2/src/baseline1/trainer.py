import os
import json
import torch
from tqdm import tqdm
from metrics import F1
from torch.utils.data import DataLoader
torch.manual_seed(42)


class Trainer:
    def __init__(self, device, trainData, validData, model, criteria, opt, batch_size, arch):
        self.device = device
        self.trainData = trainData
        self.validData = validData
        self.model = model
        self.criteria = criteria
        self.opt = opt
        self.batch_size = batch_size
        self.arch = arch
        self.history = {'train': [], 'valid': []}

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

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description, ascii=True)

        loss = 0
        f1_score = F1()

        for i, (x, y) in trange:
            o_labels, batch_loss = self.run_iter(x, y)
            if training:
                self.opt.zero_grad()
                batch_loss.backward()
                self.opt.step()

            loss += batch_loss.item()
            f1_score.update(o_labels.cpu(), y)

            trange.set_postfix(loss=loss / (i + 1), f1=f1_score.print_score(), acc=f1_score.get_accuracy())

        if training:
            self.history['train'].append(
                {'f1': f1_score.get_score(), 'accuracy': f1_score.get_accuracy(), 'loss': loss / len(trange)})
        else:
            self.history['valid'].append(
                {'f1': f1_score.get_score(), 'accuracy': f1_score.get_accuracy(), 'loss': loss / len(trange)})

    def run_iter(self, x, y):
        features = x.to(self.device)
        labels = y.to(self.device)
        o_labels = self.model(features)
        l_loss = self.criteria(o_labels, labels)
        return o_labels, l_loss

    def save(self, epoch):
        if not os.path.exists(self.arch):
            os.makedirs(self.arch)
        torch.save(self.model.state_dict(), self.arch + '/model.pkl.' + str(epoch))
        with open(self.arch + '/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
