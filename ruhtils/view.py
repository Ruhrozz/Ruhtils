"""Bunch of function to show results or samples in dataset"""


from typing import Any
import matplotlib.pyplot as plt
import torch


# TODO: add showing history
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
    """See show_dataset"""
    show_dataset(dataloader.dataset, height, width)
