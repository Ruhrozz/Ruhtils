import torch
from typing import Tuple
import numpy as np


def valid_fn(model, dataloader, criterion, device) -> Tuple[float, float]:
    model.eval()

    running_accuracy = 0
    running_loss = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predicted = model(inputs)

            loss = criterion(predicted, labels)
            running_loss += loss
            running_accuracy += np.equal(labels, predicted.argmax(axis=1)).sum()

    return running_accuracy / len(dataloader.dataset), running_loss / len(dataloader)
