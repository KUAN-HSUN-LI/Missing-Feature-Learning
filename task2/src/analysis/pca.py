import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from ipdb import set_trace as pdb
import seaborn as sns
from sklearn.decomposition import PCA


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../../data/', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.data_dir + "train.csv")
    df.drop("Id", axis=1, inplace=True)
    df.drop("Class", axis=1, inplace=True)
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)

    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)

    print('original dim: ', scaled_data.shape)
    print('after pca dim: ', x_pca.shape)

    plt.figure(figsize=(8, 6))
    df = pd.read_csv(args.data_dir + "train.csv")
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=df['Class'])
    plt.xlabel('First Principle Component')
    plt.ylabel('Second Principle Component')
    plt.savefig('PCA')


if __name__ == '__main__':
    main()
