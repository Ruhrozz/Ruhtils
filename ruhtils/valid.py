"""Model validation function"""


from typing import Tuple
from tqdm import tqdm
import torch
import numpy as np


def valid_fn(model, dataloader, criterion, **kwargs) -> Tuple[float, float]:
    """Args:
            model
            dataloader
            criterion
            kwargs:
                device
                epoch
        Return:
            (Accuracy, Loss) -  for history in view module."""
    model.eval()
    running_accuracy = 0
    running_loss = 0
    if "epoch" not in kwargs:
        kwargs["epoch"] = '???'

    pbar = tqdm(dataloader,
                desc=f"Epoch: {kwargs['epoch']} --- Validating... ",
                leave=False,
                unit="image")

    for inputs, labels in pbar:
        if "device" in kwargs:
            inputs = inputs.to(kwargs["device"])
            labels = labels.to(kwargs["device"])

        with torch.no_grad():
            predicted = model(inputs)

            loss = criterion(predicted, labels)
            running_loss += loss
            running_accuracy += np.equal(labels, predicted.argmax(axis=1)).sum()

    return running_accuracy / len(dataloader.dataset), running_loss / len(dataloader)
