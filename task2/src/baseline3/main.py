import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from preprocessor import preprocess_samples
from dataset import FeatureDataset
from model import simpleNet
from trainer import Trainer
from utils import SubmitGenerator, get_outlier
from ipdb import set_trace as pdb


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, help='architecture (model_dir)')
    parser.add_argument('--data_dir', default='../../data/', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--cuda', default=1, type=int)
    parser.add_argument('--ckpt', default=-1, type=int, help='load pre-trained model epoch')
    args = parser.parse_args()

    if args.do_train:

        dataset = pd.read_csv(args.data_dir + "train.csv")
        dataset.drop("Id", axis=1, inplace=True)

        # drop outlier
        outlier_idx = []
        features = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14']
        for f in features:
            outlier_idx += get_outlier(dataset[f])
        outlier_idx = list(set(outlier_idx))
        print(len(outlier_idx))
        dataset.drop(outlier_idx)

        train_set, valid_set = train_test_split(dataset, test_size=0.1, random_state=58)
        train = preprocess_samples(train_set, missing=["F2", "F7", "F12"])
        valid = preprocess_samples(valid_set, missing=["F2", "F7", "F12"])
        trainData = FeatureDataset(train)
        validData = FeatureDataset(valid)

        device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
        model = simpleNet(args.hidden_size)
        model.to(device)
        trainer = Trainer(device, trainData, validData, model, args.lr, args.batch_size, args.arch)

        for epoch in range(1, args.max_epoch + 1):
            print('Epoch: {}'.format(epoch))
            trainer.run_epoch(epoch, True)
            trainer.run_epoch(epoch, False)
            trainer.save(epoch)

    if args.do_predict:

        dataset = pd.read_csv(args.data_dir + "test.csv")
        dataset.drop("Id", axis=1, inplace=True)
        test = preprocess_samples(dataset, missing=["F2", "F7", "F12"])
        testData = FeatureDataset(test)

        device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
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
        for i, (x, missing, y) in trange:
            # call model.predict instead of model.forward
            o_labels = model.predict(x.to(device))
            o_labels = torch.argmax(o_labels, axis=1)
            prediction.append(o_labels.to('cpu'))

        prediction = torch.cat(prediction).detach().numpy().astype(int)
        SubmitGenerator(prediction, args.data_dir + 'sampleSubmission.csv')


if __name__ == '__main__':
    main()
