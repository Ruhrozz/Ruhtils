import torch
import numpy as np

import ruhtils.dataset as dataset


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


train, valid = dataset.make_train_valid(cfg["image_dir"],
                                        sample=0.1,
                                        extensions=("jpg",),
                                        to_rgb=True)

