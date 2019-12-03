import argparse
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# ref: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../../data/', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.data_dir + "train.csv")
    df.drop("Id", axis=1, inplace=True)
    # df.drop("Class", axis=1, inplace=True)
    corr = df.corr()

    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    plt.savefig('heat_map')


if __name__ == '__main__':
    main()
