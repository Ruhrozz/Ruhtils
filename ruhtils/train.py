"""Model training function"""


from typing import Tuple
import numpy as np


def train_fn(model, dataloader, optimizer, criterion, device) -> Tuple[float, float]:
    """Args:
            model
            dataloader
            optimizer
            criterion
            device
        Return:
            (Accuracy, Loss) -  for history in view module."""
    model.train()

    running_loss = 0
    running_accuracy = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predicted = model(inputs)
        loss = criterion(predicted, labels)

        running_loss += loss.item()
        running_accuracy += np.sum(labels == predicted.argmax(axis=1))

        loss.backward()

        optimizer.step()

    model.eval()

    return running_accuracy / len(dataloader.dataset), running_loss / len(dataloader)
