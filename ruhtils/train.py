from typing import Tuple
import numpy as np


def train_fn(model, dataloader, optimizer, criterion, device) -> Tuple[float, float]:
    model.train()

    running_loss = 0
    running_accuracy = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
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
