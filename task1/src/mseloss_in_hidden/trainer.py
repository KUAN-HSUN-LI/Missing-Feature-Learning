import os
import json
from tqdm import tqdm
from tqdm import trange
import torch
from metrics import accuracy
from torch.utils.data import DataLoader
import torch.nn as nn
from model import SimpleNet
from torch.autograd import Variable


class Trainer():
    def __init__(self, trainData, validData, hidden_size, device, model_dir="model"):
        self.history = {'train': [], 'valid': []}
        self.trainData = trainData
        self.validData = validData
        self.classficationA = SimpleNet(input_size=8, output_size=12, hidden_size=hidden_size).to(device)
        self.classficationB = SimpleNet(input_size=9, output_size=12, hidden_size=hidden_size).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.opt_C_A = torch.optim.Adam(self.classficationA.parameters(), lr=1e-4)
        self.opt_C_B = torch.optim.Adam(self.classficationB.parameters(), lr=1e-4)
        self.device = device
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_val = 0.0

    def run_epoch(self, epoch, training):
        self.classficationA.train(training)
        self.classficationB.train(training)

        if training:
            description = 'Train'
            dataset = self.trainData
            shuffle = True
        else:
            description = 'Valid'
            dataset = self.validData
            shuffle = False
        dataloader = DataLoader(dataset=dataset,
                                batch_size=256,
                                shuffle=shuffle,
                                collate_fn=dataset.collate_fn,
                                num_workers=4)

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)

        mse_loss = 0
        lossA = 0
        lossB = 0
        accA = accuracy()
        accB = accuracy()

        for i, (ft, missing_ft, labels) in trange:
            ft = ft.to(self.device)
            missing_ft = missing_ft.to(self.device)
            all_ft = torch.cat([ft, missing_ft], dim=1)
            labels = labels.to(self.device)

            # ------------------
            #  Train ClassifierA
            # ------------------

            missing_out, missing_hidden_out = self.classficationA(ft)
            all_out, all_hidden_out = self.classficationB(all_ft)
            batch_loss = self.criterion(missing_out, labels)
            batch_mse_loss = 0
            for missing_hidden, all_hidden in zip(missing_hidden_out, all_hidden_out):
                batch_mse_loss += self.mse_loss(missing_hidden, all_hidden)
            mse_loss += batch_mse_loss.item()

            if training:
                self.opt_C_A.zero_grad()
                (batch_mse_loss + batch_loss).backward()
                self.opt_C_A.step()
            lossA += batch_loss.item()
            accA.update(missing_out, labels)

            # ------------------
            #  Train ClassifierB
            # ------------------

            all_out, _ = self.classficationB(all_ft)
            batch_loss = self.criterion(all_out, labels)
            if training:
                self.opt_C_B.zero_grad()
                batch_loss.backward()
                self.opt_C_B.step()
            lossB += batch_loss.item()
            accB.update(all_out, labels)

            trange.set_postfix(accA=accA.print_score(),
                               accB=accB.print_score(),
                               lossA=lossA / (i + 1),
                               lossB=lossB / (i + 1),
                               mseLoss=mse_loss / (i + 1)
                               )
        if training:
            self.history['train'].append({
                'accA': accA.get_score(),
                'accB': accB.get_score(),
                'lossA': lossA / len(trange),
                'lossB': lossB / len(trange),
                'mseLoss': mse_loss / len(trange)})
            self.save_hist()

        else:
            self.history['valid'].append({
                'accA': accA.get_score(),
                'accB': accB.get_score(),
                'lossA': lossA / len(trange),
                'lossB': lossB / len(trange),
                'mseLoss': mse_loss / len(trange)})
            self.save_hist()
            if self.best_val < accA.get_score():
                self.best_val = accA.get_score()
                self.save_best(epoch)

    def save_best(self, epoch):
        torch.save({
            'classficationA': self.classficationA.state_dict(),
            'classficationB': self.classficationB.state_dict(),
        }, self.model_dir + '/model.pkl.'+str(epoch))

    def save_hist(self):
        with open(self.model_dir + '/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
