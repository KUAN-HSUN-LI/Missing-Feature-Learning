import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix
import numpy as np


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


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          save_path=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=[11, 11])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=6)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_history(history_path, plot_acc=True):
    """
    Ploting training process
    """
    with open(history_path, 'r') as f:
        history = json.loads(f.read())

    train_loss = [l['loss'] for l in history['train']]
    valid_loss = [l['loss'] for l in history['valid']]

    plt.figure(figsize=(7, 5))
    plt.title('Loss')
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.legend()
    plt.show()

    print('Lowest Loss ', min([[l['loss'], idx + 1] for idx, l in enumerate(history['valid'])]))

    if plot_acc:
        train_f1 = [l['acc'] for l in history['train']]
        valid_f1 = [l['acc'] for l in history['valid']]
        plt.figure(figsize=(7, 5))
        plt.title('Acc')
        plt.plot(train_f1, label='train')
        plt.plot(valid_f1, label='valid')
        plt.legend()
        plt.show()

        print('Best acc', max([[l['acc'], idx + 1] for idx, l in enumerate(history['valid'])]))
