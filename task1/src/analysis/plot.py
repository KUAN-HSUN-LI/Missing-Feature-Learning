import matplotlib.pyplot as plt
import json


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


def plot_confusion_matrix(y_true, y_pred):
    return


def plot_history(history_path):
    with open(history_path, 'r') as f:
        history = json.loads(f.read())

    train_loss = [l['loss'] for l in history['train']]
    valid_loss = [l['loss'] for l in history['valid']]
    train_f1 = [l['f1'] for l in history['train']]
    valid_f1 = [l['f1'] for l in history['valid']]

    plt.figure(figsize=(7, 5))
    plt.title('Loss')
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.title('F1 Score')
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='valid')
    plt.legend()
    plt.show()

    print('Best F1 score ', max([[l['f1'], idx] for idx, l in enumerate(history['valid'])]))
