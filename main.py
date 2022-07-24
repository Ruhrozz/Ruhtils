import torch
import numpy as np
import torchvision
import ruhtils.dataset as dataset
import ruhtils.transforms as transforms
import matplotlib.pyplot as plt


cfg = {
    "backbone": "resnet50",
    "pretrained": False,
    "num_classes": 2,

    "image_dir": "datasets/pizza_not_pizza/",

    "val_size": 0.1,

    "batch_size": 3,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "epoch_size": 5,
}


train_set = dataset.ImageFolder(cfg["image_dir"],
                                use_albumentations=True,
                                transform=transforms.get_transforms(is_train=True))

valid_set = dataset.Dataset(root=cfg["image_dir"],
                            samples=train_set.take_valid(),
                            use_albumentations=True,
                            transform=transforms.get_transforms())

plt.imshow(valid_set[0][0].permute(1, 2, 0))
plt.show()
