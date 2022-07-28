import ruhtils.data.dataset as dataset
import ruhtils.data.dataloader as dataloader
import ruhtils.data.transforms as transforms
import ruhtils.view as view
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
                            samples=train_set.take_valid(sample=0.005),
                            use_albumentations=True,
                            transform=transforms.get_transforms())


valid_loader = dataloader.get_dataloader(valid_set, device=cfg["device"], **cfg["dataloader_cfg"])

print(len(valid_set))
view.show_dataloader(valid_loader)
