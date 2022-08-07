"""Bunch of function to show results or samples in dataset"""


from typing import Any
import matplotlib.pyplot as plt
import torch


# TODO: add showing image for each label
# TODO: add showing right and wrong answers
def show_dataset(dataset: Any, height: int = 3, width: int = 4) -> None:
    """Shows image consisting of Height x Width dataset's samples
    Args:
        dataset
        height
        width
    """
    if len(dataset) < height * width:
        height = width = round(len(dataset) ** 0.5)

    images = [dataset[i][0] for i in range(height * width)]

    size = dataset[0][0].shape

    # Best regards to AetelFinch
    if len(images):
        image = torch.cat(images, dim=2)
        image = torch.split(image, size[2] * width, dim=2)
        image = torch.hstack(image)

        plt.imshow(image.permute([1, 2, 0]))
        plt.axis('off')
        plt.show()


def show_dataloader(dataloader: Any, height: int = 3, width: int = 4) -> None:
    """Same as show_dataset"""
    show_dataset(dataloader.dataset, height, width)


def show_plots(train_history, valid_history, save=False):
    """Create accuracy and loss plots.
    Args shape:
        [[Acc1  Acc2  ... AccN]
         [Loss1 Loss2 ... LossN]]
    Args:
        train_history: ndarray
        valid_history: ndarray
        save:
            Save or show image.
    """
    assert train_history.shape[0] == 2
    assert train_history.shape[0] == 2
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(train_history[0], "r")
    plt.plot(valid_history[0], "g--")
    plt.grid()
    if save:
        plt.savefig("Accuracy.png")
        plt.close()
    else:
        plt.show()

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_history[1], "r")
    plt.plot(valid_history[1], "g--")
    plt.grid()
    if save:
        plt.savefig("Loss.png")
        plt.close()
    else:
        plt.show()
