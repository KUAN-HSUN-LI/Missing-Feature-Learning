import os
import json
import torch
from tqdm import tqdm
from metrics import Accuracy
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from models import Generator, Discriminator, SimpleNet
torch.manual_seed(42)
from ipdb import set_trace as pdb


class Trainer:
    def __init__(self, device, trainData, validData, hidden_size, lr, batch_size, arch):
        self.device = device
        self.trainData = trainData
        self.validData = validData
        self.model = SimpleNet(hidden_size).to(device)
        self.generator = Generator(hidden_size).to(device)
        self.discriminator = Discriminator(hidden_size).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.scheduler = StepLR(self.opt, step_size=150, gamma=0.5)
        self.scheduler_G = StepLR(self.opt_G, step_size=300, gamma=0.5)
        self.scheduler_D = StepLR(self.opt_D, step_size=300, gamma=0.5)
        self.batch_size = batch_size
        self.arch = arch
        self.history = {'train': [], 'valid': []}

    def run_epoch(self, epoch, training):
        self.model.train(training)
        self.generator.train(training)
        self.discriminator.train(training)

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

        g_loss = 0
        d_loss = 0
        loss = 0
        accuracy = Accuracy()

        for i, (features, real_missing, labels) in trange:

            features = features.to(self.device)             # (batch, 11)
            real_missing = real_missing.to(self.device)     # (batch, 3)
            labels = labels.to(self.device)                 # (batch, 1)
            batch_size = features.shape[0]

            if training:
                rand = torch.rand((batch_size, 11)).to(self.device) - 0.5
                std = features.std(dim=1)
                noise = rand * std.unsqueeze(1)
                features += noise

            # Adversarial ground truths
            valid = torch.FloatTensor(batch_size, 1).fill_(1.0).to(self.device)     # (batch, 1)
            fake = torch.FloatTensor(batch_size, 1).fill_(0.0).to(self.device)      # (batch, 1)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            if i % 10 < 5 or not training:
                real_pred = self.discriminator(real_missing)
                d_real_loss = self.criterion(real_pred, valid)

                fake_missing = self.generator(features.detach())
                fake_pred = self.discriminator(fake_missing)
                d_fake_loss = self.criterion(fake_pred, fake)
                batch_d_loss = (d_real_loss + d_fake_loss)

                if training:
                    self.opt_D.zero_grad()
                    batch_d_loss.backward()
                    self.opt_D.step()
                d_loss += batch_d_loss.item()

            # -----------------
            #  Train Generator
            # -----------------

            if i % 10 >= 5 or not training:
                gen_missing = self.generator(features.detach())
                validity = self.discriminator(gen_missing)
                batch_g_loss = self.criterion(validity, valid)

                if training:
                    self.opt_G.zero_grad()
                    batch_g_loss.backward()
                    self.opt_G.step()
                g_loss += batch_g_loss.item()

                # ------------------
                #  Train Classifier
                # ------------------

                gen_missing = self.generator(features.detach())
                all_features = torch.cat((features, gen_missing), dim=1)
                o_labels = self.model(all_features)
                batch_loss = self.criterion(o_labels, labels)

                if training:
                    self.opt.zero_grad()
                    batch_loss.backward()
                    self.opt.step()
                loss += batch_loss.item()
                accuracy.update(o_labels, labels)

                trange.set_postfix(accuracy=accuracy.print_score(),
                                   g_loss=g_loss / (i + 1),
                                   d_loss=d_loss / (i + 1),
                                   loss=loss / (i + 1))

        if training:
            self.history['train'].append({
                'accuracy': accuracy.get_score(), 'g_loss': g_loss / len(trange), 'd_loss': d_loss / len(trange), 'loss': loss / len(trange)})
            self.scheduler.step()
            self.scheduler_G.step()
            self.scheduler_D.step()
        else:
            self.history['valid'].append({
                'accuracy': accuracy.get_score(), 'g_loss': g_loss / len(trange), 'd_loss': d_loss / len(trange), 'loss': loss / len(trange)})

    def save(self, epoch):
        if not os.path.exists(self.arch):
            os.makedirs(self.arch)

        path = self.arch + '/model.pkl.' + str(epoch)
        torch.save({
            'model': self.model.state_dict(),
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, path)
        with open(self.arch + '/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
