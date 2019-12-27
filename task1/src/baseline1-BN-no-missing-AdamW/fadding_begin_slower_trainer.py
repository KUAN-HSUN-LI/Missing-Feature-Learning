import os
import json
from tqdm import tqdm
from tqdm import trange
import torch
from metrics import accuracy
from torch.utils.data import DataLoader
from model import SimpleNet


class Trainer():
    def __init__(self, device, trainData, validData, args):
        self.device = device
        self.history = {'train': [], 'valid': []}
        self.trainData = trainData
        self.validData = validData

        self.fadding_model = SimpleNet(input_size=9, output_size=12,
                                       hidden_size=args.hidden_size).to(device)
        self.fadding_model.load_state_dict(torch.load("model0.33/model.pkl.904"))
        self.fixed_model = SimpleNet(input_size=9, output_size=12,
                                     hidden_size=args.hidden_size).to(device)
        self.fixed_model.load_state_dict(torch.load("model0.33/model.pkl.904"))

        self.criteria = torch.nn.MSELoss()
        self.opt = torch.optim.AdamW(self.fadding_model.parameters(), lr=8e-5, weight_decay=9e-3)
        # self.scheduler = scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=200, gamma=args.step_lr)
        self.batch_size = args.batch_size
        self.model_dir = args.arch
        self.decay_value = 1
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_val = 0.0

    def run_epoch(self, epoch, training):
        self.fadding_model.train(training)
        self.fixed_model.train(False)

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
        acc_fadding = accuracy()
        acc_fixed = accuracy()
        self.decay_value *= (0.99 / (1 + epoch / 10000))
        for i, (ft, missing_ft, labels) in trange:
            ft = ft.to(self.device)
            missing_ft = missing_ft.to(self.device)
            labels = labels.to(self.device)

            missing_fadding_ft = missing_ft * self.decay_value
            missing_0_ft = missing_ft * 0

            fadding_ft = torch.cat([missing_fadding_ft, ft], dim=1)
            zero_ft = torch.cat([missing_0_ft, ft], dim=1)
            raw_ft = torch.cat([missing_ft, ft], dim=1)

            fadding_out, fadding_hiddens = self.fadding_model(fadding_ft)
            zero_out, _ = self.fadding_model(zero_ft)
            raw_out, raw_hiddens = self.fixed_model(raw_ft)

            batch_loss = 0
            for raw_hidden, fadding_hidden in zip(raw_hiddens, fadding_hiddens):
                batch_loss += self.criteria(raw_hidden, fadding_hidden)

            batch_loss += self.criteria(raw_out, fadding_out)

            if training:
                self.opt.zero_grad()
                batch_loss.backward()
                self.opt.step()

            loss += batch_loss.item()
            acc_fadding.update(fadding_out, labels)
            acc_fixed.update(zero_out, labels)

            trange.set_postfix(
                loss=loss / (i + 1), acc_fadding=acc_fadding.print_score(), acc_fixed=acc_fixed.print_score())

        # self.scheduler.step()

        if training:
            self.history['train'].append(
                {'acc-fadding': acc_fadding.get_score(), 'acc_fixed': acc_fixed.get_score(), 'loss': loss / len(trange)})
            self.save_hist()
        else:
            self.history['valid'].append(
                {'acc-fadding': acc_fadding.get_score(), 'acc_fixed': acc_fixed.get_score(), 'loss': loss / len(trange)})
            self.save_hist()
            if acc_fixed.get_score() > self.best_val:
                self.best_val = acc_fixed.get_score()
                self.save_best(epoch)

    def run_iter(self, x, y):
        features = x.to(self.device)
        labels = y.to(self.device)
        o_labels, hiddens = self.model(features)
        l_loss = self.criteria(o_labels, labels)
        return o_labels, l_loss

    def save_best(self, epoch):
        torch.save(self.fadding_model.state_dict(), self.model_dir + '/model.pkl.'+str(epoch))

    def save_hist(self):
        with open(self.model_dir + '/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
