import os
import json
import torch
from tqdm import tqdm
from metrics import Accuracy
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
torch.manual_seed(42)
from ipdb import set_trace as pdb


class Trainer:
    def __init__(self, device, trainData, validData, model, criteria, opt, batch_size, arch):
        self.device = device
        self.trainData = trainData
        self.validData = validData
        self.model = model
        self.criteria = criteria
        self.opt = opt
        self.scheduler = StepLR(self.opt, step_size=100, gamma=0.5)
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
        accuracy = Accuracy()

        for i, (x, y) in trange:
            o_labels, batch_loss = self.run_iter(x, y)
            if training:
                self.opt.zero_grad()
                batch_loss.backward()
                self.opt.step()

            loss += batch_loss.item()
            accuracy.update(o_labels.cpu(), y)

            trange.set_postfix(accuracy=accuracy.print_score(), loss=loss / (i + 1))

        if training:
            self.history['train'].append({'accuracy': accuracy.get_score(), 'loss': loss / len(trange)})
            self.scheduler.step()
        else:
            self.history['valid'].append({'accuracy': accuracy.get_score(), 'loss': loss / len(trange)})

    def run_iter(self, x, y):
        features = x.to(self.device)
        labels = y.to(self.device)
        o_labels = self.model(features)
        l_loss = self.criteria(o_labels, labels)
        return o_labels, l_loss

    def save(self, epoch):
        if not os.path.exists(self.arch):
            os.makedirs(self.arch)
        if epoch % 10 == 0:
            torch.save(self.model.state_dict(), self.arch + '/model.pkl.' + str(epoch))
            with open(self.arch + '/history.json', 'w') as f:
                json.dump(self.history, f, indent=4)
