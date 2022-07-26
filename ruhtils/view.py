import matplotlib.pyplot as plt
from typing import Any
import torch


def show_dataset(dataset: Any, height: int = 3, width: int = 4) -> None:
    images = [dataset[i][0] for i in range(height * width)]
    size = images[0].shape

    # Best regards to AetelFinch
    image = torch.cat(images, dim=2)
    image = torch.split(image, size[2] * width, dim=2)
    image = torch.hstack(image)

    plt.imshow(image.permute([1, 2, 0]))
    plt.axis('off')
    plt.show()


def show_dataloader(dataloader: Any, height: int = 3, width: int = 4) -> None:
    show_dataset(dataloader.dataset, height, width)





























