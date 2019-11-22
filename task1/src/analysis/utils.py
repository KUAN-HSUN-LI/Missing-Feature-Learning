import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


def counter(data):
    keys = Counter(data).keys()
    values = Counter(data).values()
    key_df = pd.DataFrame(data=keys, columns=["key"])
    value_df = pd.DataFrame(data=values, columns=["quntity"])
    percent_df = pd.DataFrame([i / len(data) * 100.0 for i in values], columns=["percentage"])

    df = pd.concat([key_df, value_df, percent_df], axis=1)

    print(df.round(3))


def plot_box(data, y_ticks=False, save_path=None):
    length = len(data)
    fig, ax = plt.subplots(1, length)
    for idx, d in enumerate(data):
        ax[idx].boxplot(d)
        ax[idx].set_xticks([])
        ax[idx].set_xlabel(str(idx+1))
        if not y_ticks:
            ax[idx].set_yticks([])
        else:
            plt.subplots_adjust(wspace=1.5, hspace=1)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


def get_outlier(data):
    tmp = np.percentile(data, (25, 50, 75), interpolation='midpoint')
    Q1 = tmp[0]
    Q2 = tmp[1]
    Q3 = tmp[2]
    QD = (Q3 - Q1) / 2
    max_value = Q3 + QD * 3
    min_value = Q1 - QD * 3
    print(Q2, max_value, min_value)
    outlier_idx = []

    for idx, d in enumerate(data):
        if d > max_value or d < min_value:
            outlier_idx.append(idx)
    return outlier_idx
