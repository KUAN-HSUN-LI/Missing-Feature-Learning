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
    def __init__(self, device, trainData, validData, model, lr, batch_size, arch):
        self.device = device
        self.trainData = trainData
        self.validData = validData
        self.model = model
        self.criteria = torch.nn.CrossEntropyLoss()
        self.missing_criteria = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.opt, step_size=100, gamma=0.1)
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

        f_loss = 0
        l_loss = 0
        accuracy = Accuracy()

        for i, (x, missing, y) in trange:
            o_labels, batch_f_loss, batch_l_loss = self.run_iter(x, missing, y)
            batch_loss = batch_f_loss + batch_l_loss

            if training:
                self.opt.zero_grad()
                batch_loss.backward()
                self.opt.step()

            f_loss += batch_f_loss.item()
            l_loss += batch_l_loss.item()
            accuracy.update(o_labels.cpu(), y)

            trange.set_postfix(accuracy=accuracy.print_score(), f_loss=f_loss / (i + 1), l_loss=l_loss / (i + 1))

        if training:
            self.history['train'].append({'accuracy': accuracy.get_score(), 'loss': f_loss / len(trange)})
            self.scheduler.step()
        else:
            self.history['valid'].append({'accuracy': accuracy.get_score(), 'loss': f_loss / len(trange)})

    def run_iter(self, x, missing, y):
        features = x.to(self.device)
        missing = missing.to(self.device)
        labels = y.to(self.device)
        o_missing, o_labels = self.model(features)
        f_loss = self.missing_criteria(o_missing, missing)
        l_loss = self.criteria(o_labels, labels.argmax(dim=1))
        return o_labels, f_loss, l_loss

    def save(self, epoch):
        if not os.path.exists(self.arch):
            os.makedirs(self.arch)
        if epoch % 10 == 0:
            torch.save(self.model.state_dict(), self.arch + '/model.pkl.' + str(epoch))
            with open(self.arch + '/history.json', 'w') as f:
                json.dump(self.history, f, indent=4)
