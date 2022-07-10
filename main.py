import numpy as np
import matplotlib.pyplot as plt
import torch


def show_samples(dataloader, items, n_samples=9):
    # It shows `n` samples of dataset's images
    # and writes label on bottom of the image

    if n_samples <= 0:
        return

    plt.rcParams["figure.figsize"] = (15, 15)

    grid = np.ceil(np.sqrt(n_samples)).astype(int)
    images, labels = next(iter(dataloader))

    jump = 0

    for i in range(n_samples):
        if i + jump >= len(images):
            images, labels = next(iter(dataloader))
            jump -= i

        plt.subplot(grid, grid, i + 1)
        plt.imshow(torch.permute(images[i + jump], [1, 2, 0]))
        plt.xlabel(items[labels[i + jump].item()])

    plt.show()

