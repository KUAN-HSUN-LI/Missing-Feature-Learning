import os
import json
from tqdm.notebook import tqdm
from tqdm import trange
import torch
from metrics import accuracy
from torch.utils.data import DataLoader
import torch.nn as nn
from model import Generator, Discriminator, SimpleNet
from torch.autograd import Variable


class Trainer():
    def __init__(self, trainData, validData, hidden_size, device, model_dir="model"):
        self.history = {'train': [], 'valid': []}
        self.trainData = trainData
        self.validData = validData
        self.generator = Generator(input_size=8, output_size=1, hidden_size=hidden_size).to(device)
        self.discriminator = Discriminator(input_size=1, output_size=1, hidden_size=hidden_size).to(device)
        self.classfication = SimpleNet(input_size=9, output_size=12, hidden_size=hidden_size).to(device)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.opt_C = torch.optim.Adam(self.classfication.parameters(), lr=1e-4)
        self.device = device
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_val = 0.0

    def run_epoch(self, epoch, training):
        self.generator.train(training)
        self.discriminator.train(training)
        self.classfication.train(training)

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

        g_loss = 0
        d_loss = 0
        loss = 0
        acc = accuracy()

        for i, (ft, missing_ft, labels) in trange:
            ft = ft.to(self.device)
            missing_ft = missing_ft.to(self.device)
            labels = labels.to(self.device)
            batch_size = ft.shape[0]
            true = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(self.device)     # (batch, 1)
            fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(self.device)     # (batch, 1)

            # -----------------
            #  Train Generator
            # -----------------

            gen_missing = self.generator(ft.detach())
            validity = self.discriminator(gen_missing)
            batch_g_loss = self.adversarial_loss(validity, true)

            if training:
                self.opt_G.zero_grad()
                batch_g_loss.backward()
                self.opt_G.step()
            g_loss += batch_g_loss.item()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            real_pred = self.discriminator(missing_ft)
            d_real_loss = self.adversarial_loss(real_pred, true)

            fake_missing = self.generator(ft.detach())
            fake_pred = self.discriminator(fake_missing)
            d_fake_loss = self.adversarial_loss(fake_pred, fake)
            batch_d_loss = (d_real_loss + d_fake_loss) / 2

            if training:
                self.opt_D.zero_grad()
                batch_d_loss.backward()
                self.opt_D.step()
            d_loss += batch_d_loss.item()

            # ------------------
            #  Train Classifier
            # ------------------

            gen_missing = self.generator(ft.detach())
            all_features = torch.cat((ft, gen_missing), dim=1)
            o_labels = self.classfication(all_features)
            batch_loss = self.criterion(o_labels, labels)
            if training:
                self.opt_C.zero_grad()
                batch_loss.backward()
                self.opt_C.step()
            loss += batch_loss.item()

            acc.update(o_labels, labels)

            trange.set_postfix(acc=acc.print_score(),
                               g_loss=g_loss / (i + 1),
                               d_loss=d_loss / (i + 1),
                               loss=loss / (i + 1))

        if training:
            self.history['train'].append({
                'acc': acc.get_score(), 'g_loss': g_loss / len(trange), 'd_loss': d_loss / len(trange), 'loss': loss / len(trange)})
            self.save_hist()

            # self.scheduler.step()
            # self.scheduler_G.step()
            # self.scheduler_D.step()
        else:
            self.history['valid'].append({
                'acc': acc.get_score(), 'g_loss': g_loss / len(trange), 'd_loss': d_loss / len(trange), 'loss': loss / len(trange)})
            self.save_hist()
            if self.best_val < acc.get_score():
                self.best_val = acc.get_score()
                self.save_best(epoch)

    def save_best(self, epoch):
        torch.save({
            'cls': self.classfication.state_dict(),
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, self.model_dir + '/model.pkl.'+str(epoch))

    def save_hist(self):
        with open(self.model_dir + '/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
