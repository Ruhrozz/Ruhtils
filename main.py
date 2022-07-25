import torch
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import ruhtils.dataset as dataset
import ruhtils.transforms as transforms
import ruhtils.model as net
import matplotlib.pyplot as plt


cfg = {
    "backbone": "resnet50",
    "num_classes": 2,
    "weights": None,

    "image_dir": "datasets/pizza_not_pizza/",
    "val_size": 0.1,

    "batch_size": 3,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "epoch_size": 5,
}

model = net.get_model(cfg["backbone"],
                      cfg["num_classes"],
                      weights=cfg["weights"])

train_set = dataset.ImageFolder(cfg["image_dir"],
                                use_albumentations=True,
                                transform=transforms.get_transforms(is_train=True))

valid_set = dataset.Dataset(root=cfg["image_dir"],
                            samples=train_set.take_valid(),
                            use_albumentations=True,
                            transform=transforms.get_transforms())

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# print(train_loader.dataset)
# print(len(train_set))
