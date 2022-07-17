import numpy as np
import matplotlib.pyplot as plt
import torch


names = {0: "Not pizza", 1: "Pizza"}


def show_samples(dataloader, items=None, n_samples=9):
    """ It shows `n` samples of dataset's images
    and writes label on bottom of the image.
    
    Parameters
    ----------
    dataloader: torch.utils.data.dataloader.DataLoader
        Where will pictures be taken from.
        
    items: dict
        Names of items on pictures.
        
    n_samples: int
        How much pictures to show on the screen.
        
    """"
    
    
    if n_samples <= 0:
        return 

    plt.rcParams["figure.figsize"] = (13, 13)

    grid = np.ceil(np.sqrt(n_samples)).astype(int)
    images, labels = next(iter(dataloader))

    jump = 0

    for i in range(n_samples):
        if i + jump >= len(images):
            images, labels = next(iter(dataloader))
            jump = -i

        plt.subplot(grid, grid, i + 1)
        plt.imshow(torch.permute(images[i + jump], [1, 2, 0]))
        if items is not None:
            plt.xlabel(items[labels[i + jump].item()])
        else:
            plt.axis('off')


    plt.show()

show_samples(dataloader, n_samples=4)

