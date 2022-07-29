"""Module keeps functions for getting models"""


from typing import Optional
import torch


def _get_pretrained_resnet50(weights, num_classes):
    model = torch.hub.load('pytorch/vision',
                           model='resnet50',
                           weights=weights)

    model.fc.out_features = num_classes

    return model


def _get_pretrained_mobilenet_v2(weights, num_classes):
    model = torch.hub.load('pytorch/vision',
                           model='mobilenet_v2',
                           weights=weights)

    model.classifier[1].out_features = num_classes

    return model


# TODO: simplify protected get-functions
def get_model(backbone: str,
              num_classes: int,
              weights: Optional[str] = None):
    """Function takes last version of model from GitHub.
    Args:
        backbone: str
            Name of model.
        weights: Optional[str]
            If model is pretrained, write training dataset.
            It is a new torch feature, see their GitHub.
        num_classes: int
    Return:
        model
    """

    if weights is not None:
        if backbone == "resnet50":
            return _get_pretrained_resnet50(weights, num_classes)
        if backbone == "mobilenet_v2":
            return _get_pretrained_mobilenet_v2(weights, num_classes)

    model = torch.hub.load('pytorch/vision',
                           model=backbone,
                           weights=weights,
                           num_classes=num_classes)
    return model
