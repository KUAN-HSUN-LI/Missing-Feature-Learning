import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from preprocessor import preprocess_samples
from dataset import FeatureDataset
from model import simpleNet
from trainer import Trainer
from utils import SubmitGenerator, get_outlier


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, help='architecture (model_dir)')
    parser.add_argument('--data_dir', default='../../data/', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--ckpt', default=-1, type=int, help='load pre-trained model epoch')
    args = parser.parse_args()

    if args.do_train:

        dataset = pd.read_csv(args.data_dir + "train.csv")
        dataset.drop("Id", axis=1, inplace=True)
        dataset.drop("F2", axis=1, inplace=True)
        dataset.drop("F7", axis=1, inplace=True)
        dataset.drop("F12", axis=1, inplace=True)
        train_set, valid_set = train_test_split(dataset, test_size=0.1, random_state=58)
        train = preprocess_samples(train_set, missing=["F2", "F7", "F12"])
        valid = preprocess_samples(valid_set, missing=["F2", "F7", "F12"])
        trainData = FeatureDataset(train)
        validData = FeatureDataset(valid)

        device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
        model = simpleNet(args.hidden_size)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        # criteria = torch.nn.CrossEntropyLoss()
        criteria = torch.nn.BCEWithLogitsLoss()
        max_epoch = args.max_epoch
        batch_size = args.batch_size
        trainer = Trainer(device, trainData, validData, model, criteria, opt, batch_size, args.arch)

        for epoch in range(1, max_epoch + 1):
            print('Epoch: {}'.format(epoch))
            trainer.run_epoch(epoch, True)
            trainer.run_epoch(epoch, False)
            trainer.save(epoch)

    if args.do_predict:

        dataset = pd.read_csv(args.data_dir + "test.csv")
        dataset.drop("Id", axis=1, inplace=True)
        dataset.drop("F2", axis=1, inplace=True)
        dataset.drop("F7", axis=1, inplace=True)
        dataset.drop("F12", axis=1, inplace=True)
        test = preprocess_samples(dataset, missing=["F2", "F7", "F12"])
        testData = FeatureDataset(test)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = simpleNet(args.hidden_size)
        model.load_state_dict(torch.load('%s/model.pkl.%d' % (args.arch, args.ckpt)))
        model.train(False)
        model.to(device)
        dataloader = DataLoader(dataset=testData,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=testData.collate_fn,
                                num_workers=4)
        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
        prediction = []
        for i, (x, y) in trange:
            o_labels = model(x.to(device))
            o_labels = F.sigmoid(o_labels) > 0.5
            prediction.append(o_labels.to('cpu'))

        prediction = torch.cat(prediction).detach().numpy().astype(int)
        SubmitGenerator(prediction, args.data_dir + 'sampleSubmission.csv')


if __name__ == '__main__':
    main()
