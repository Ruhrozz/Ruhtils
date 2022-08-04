"""
This module is an example of usage ruhtils package.
You can find here:
1. Creating dataset and dataloader
2. Augmentations using albumentations
3. Training model such as "ResNet50"
"""

# import time
import numpy as np
# from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.nn import CrossEntropyLoss

from ruhtils.data import dataset
from ruhtils.data import dataloader
from ruhtils.data import transforms
from ruhtils.train import train_fn
from ruhtils.valid import valid_fn
# from ruhtils import view
import ruhtils.model as net


cfg = {
    "backbone": "mobilenet_v2",
    "num_classes": 2,
    "weights": "IMAGENET1K_V1",

    "image_dir": "datasets/pizza_not_pizza/",
    "val_size": 0.1,

    "device": "CPU",

    "dataloader_cfg": {
        "batch_size": 3,
        "shuffle": True,
        "num_workers": 0,
        "drop_last": False,
    },

    "optimizer_cfg": {
        "lr": 1e-3,
        "weight_decay": 1e-2,

    },

    "epoch_size": 5,
}

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

model = net.get_model(cfg["backbone"],
                      cfg["num_classes"],
                      weights=cfg["weights"])


train_set = dataset.ImageFolder(cfg["image_dir"],
                                # use_albumentations=True,
                                # transform=transforms.kbrodt_transforms(is_train=True),
                                transform=transforms.torch_transforms(is_train=True),
                                )
valid_set = dataset.Dataset(samples=train_set.take_valid(sample=0.005),
                            # use_albumentations=True,
                            # transform=transforms.kbrodt_transforms(),
                            transform=transforms.torch_transforms(),
                            )

train_loader = dataloader.get_dataloader(train_set, device=cfg["device"], **cfg["dataloader_cfg"])
valid_loader = dataloader.get_dataloader(valid_set, device=cfg["device"], **cfg["dataloader_cfg"])


optimizer = optim.Adam(model.parameters(), **cfg["optimizer_cfg"])
criterion = CrossEntropyLoss()

v_history = t_history = np.array([0, 0])

for epoch in range(cfg["epoch_size"]):
    t_h_train = train_fn(model,
                         valid_loader,
                         optimizer,
                         criterion,
                         device=device,
                         epoch=epoch)

    t_h_valid = valid_fn(model,
                         valid_loader,
                         criterion,
                         device=device,
                         epoch=epoch)

    t_history = np.vstack((t_history, t_h_train))
    v_history = np.vstack((v_history, t_h_valid))

plt.plot(t_history.T[0], "r")
plt.plot(t_history.T[1], "g--")
plt.grid()
plt.show()
