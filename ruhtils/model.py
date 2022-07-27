import torch
from typing import Optional, Any


# TODO: "get pretrained ResNet" function
# TODO: "get pretrained MobileNetV2" function


def get_model(backbone: str,
              num_classes: int,
              weights: Optional[Any] = None):

    model = torch.hub.load('pytorch/vision',
                           model=backbone,
                           weights=weights,
                           num_classes=num_classes)
    return model
