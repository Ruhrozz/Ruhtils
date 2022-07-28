import torch
from typing import Optional, Any


def get_pretrained_resnet50(weights, num_classes):
    model = torch.hub.load('pytorch/vision',
                           model='resnet50',
                           weights=weights)

    model.fc.out_features = num_classes

    return model


def get_pretrained_mobilenet_v2(weights, num_classes):
    model = torch.hub.load('pytorch/vision',
                           model='mobilenet_v2',
                           weights=weights)

    model.classifier[1].out_features = num_classes

    return model


def get_model(backbone: str,
              num_classes: int,
              weights: Optional[Any] = None):

    if weights is not None:
        if backbone == "resnet50":
            return get_pretrained_resnet50(weights, num_classes)
        elif backbone == "mobilenet_v2":
            return get_pretrained_mobilenet_v2(weights, num_classes)

    model = torch.hub.load('pytorch/vision',
                           model=backbone,
                           weights=weights,
                           num_classes=num_classes)
    return model
