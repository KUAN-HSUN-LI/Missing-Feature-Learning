from utils import SubmitGenerator
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from preprocessor import preprocess_samples
from dataset import FeatureDataset
from fadding_begin_slower_trainer import Trainer
from plot import plot_history
from model import SimpleNet
torch.manual_seed(42)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default="model", help='architecture (model_dir)')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--max_epoch', default=10000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--step_lr', default=0.5, type=float)
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--ckpt', type=int, help='load pre-trained model epoch')
    args = parser.parse_args()

    if args.do_train:

        dataset = pd.read_csv("../../data/train.csv")
        dataset.drop("Id", axis=1, inplace=True)
        train_set, valid_set = train_test_split(dataset, test_size=0.1, random_state=73)
        feature_for_training = ["F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]
        feature_for_prediction = ["F1"]

        train = preprocess_samples(train_set, feature_for_training, feature_for_prediction)
        valid = preprocess_samples(valid_set, feature_for_training, feature_for_prediction)

        trainData = FeatureDataset(train)
        validData = FeatureDataset(valid)

        device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
        max_epoch = args.max_epoch
        trainer = Trainer(device, trainData, validData, args)

        for epoch in range(1, max_epoch + 1):
            print('Epoch: {}'.format(epoch))
            trainer.run_epoch(epoch, True)
            trainer.run_epoch(epoch, False)

    if args.do_predict:

        dataset = pd.read_csv("../../data/test.csv")
        dataset.drop("Id", axis=1, inplace=True)
        feature_for_testing = ["F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]
        test = preprocess_samples(dataset, feature_for_testing)

        testData = FeatureDataset(test)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleNet(input_size=9, output_size=12, hidden_size=args.hidden_size)
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
        for i, (ft, _, y) in trange:
            b = ft.shape[0]
            missing_ft = torch.zeros(b, 1)
            all_ft = torch.cat([missing_ft, ft], dim=1)
            o_labels, _ = model(all_ft.to(device))
            o_labels = torch.argmax(o_labels, axis=1)
            prediction.append(o_labels.to('cpu').numpy().tolist())

        prediction = sum(prediction, [])
        SubmitGenerator(prediction, "../../data/sampleSubmission.csv")

    if args.do_plot:
        plot_history("{file}/history.json".format(file=args.arch))


if __name__ == '__main__':
    main()
