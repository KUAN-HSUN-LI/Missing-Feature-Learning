import os
import json
from tqdm import tqdm
from tqdm import trange
import torch
from f1_score import F1
from torch.utils.data import DataLoader


class Trainer():
    def __init__(self, trainData, validData, model, criteria, opt, model_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {'train': [], 'valid': []}
        self.trainData = trainData
        self.validData = validData
        self.model = model.to(self.device)
        self.criteria = criteria
        self.opt = opt
        self.model_dir = model_dir

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
                                batch_size=256,
                                shuffle=shuffle,
                                collate_fn=dataset.collate_fn,
                                num_workers=4)

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)

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

            trange.set_postfix(
                loss=loss / (i + 1), f1=f1_score.print_score())
        if training:
            self.history['train'].append({'f1': f1_score.get_score(), 'loss': loss / len(trange)})
        else:
            self.history['valid'].append({'f1': f1_score.get_score(), 'loss': loss / len(trange)})

    def run_iter(self, x, y):
        features = x.to(self.device)
        labels = y.to(self.device)
        o_labels = self.model(features)
        l_loss = self.criteria(o_labels, labels)
        return o_labels, l_loss

    def save(self, epoch):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.model.state_dict(), self.model_dir + '/model.pkl.'+str(epoch))
        with open(self.model_dir + '/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
