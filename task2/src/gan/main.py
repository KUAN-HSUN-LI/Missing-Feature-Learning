import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from preprocessor import preprocess_samples
from dataset import FeatureDataset
from trainer import Trainer
from utils import SubmitGenerator, get_outlier
from models import Generator, Discriminator, SimpleNet
from ipdb import set_trace as pdb


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, help='architecture (model_dir)')
    parser.add_argument('--data_dir', default='../../data/', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=800, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--cuda', default=1, type=int)
    parser.add_argument('--ckpt', default=-1, type=int, help='load pre-trained model epoch')
    args = parser.parse_args()

    if args.do_train:

        dataset = pd.read_csv(args.data_dir + "train.csv")
        dataset.drop("Id", axis=1, inplace=True)

        train_set, valid_set = train_test_split(dataset, test_size=0.2, random_state=42)
        train = preprocess_samples(train_set, missing=["F2", "F7", "F12"])
        valid = preprocess_samples(valid_set, missing=["F2", "F7", "F12"])
        trainData = FeatureDataset(train)
        validData = FeatureDataset(valid)

        device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
        trainer = Trainer(device, trainData, validData, args.hidden_size, args.lr, args.batch_size, args.arch)

        for epoch in range(1, args.max_epoch + 1):
            print('Epoch: {}'.format(epoch))
            trainer.run_epoch(epoch, True)
            trainer.run_epoch(epoch, False)
            if epoch % 50 == 0:
                trainer.save(epoch)

    if args.do_predict:

        dataset = pd.read_csv(args.data_dir + "test.csv")
        dataset.drop("Id", axis=1, inplace=True)
        test = preprocess_samples(dataset, missing=["F2", "F7", "F12"])
        testData = FeatureDataset(test)

        path = '%s/model.pkl.%d' % (args.arch, args.ckpt)
        checkpoint = torch.load(path)
        device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')

        model = SimpleNet(args.hidden_size)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.train(False)
        generator = Generator(args.hidden_size)
        generator.load_state_dict(checkpoint['generator'])
        generator.to(device)
        generator.train(False)

        dataloader = DataLoader(dataset=testData,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=testData.collate_fn,
                                num_workers=4)
        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
        prediction = []
        for i, (features, missing, y) in trange:

            gen_missing = generator(features.to(device))
            all_features = torch.cat((features.to(device), gen_missing.to(device)), dim=1)
            o_labels = model(all_features)
            o_labels = F.sigmoid(o_labels) > 0.5
            prediction.append(o_labels.to('cpu'))

        prediction = torch.cat(prediction).detach().numpy().astype(int)
        SubmitGenerator(prediction, args.data_dir + 'sampleSubmission.csv')


if __name__ == '__main__':
    main()
