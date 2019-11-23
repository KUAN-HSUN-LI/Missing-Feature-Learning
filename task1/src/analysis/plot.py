import matplotlib.pyplot as plt


def plot_box(data, y_ticks=False, save_path=None):
    """
    Args: 
        data (list): shape = (number of features, data's length)
    """
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


def plot_confusion_matrix():
    return
