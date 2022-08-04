"""Model training function"""


from typing import Tuple
from tqdm import tqdm


def train_fn(model, dataloader, optimizer, criterion, **kwargs) -> Tuple[float, float]:
    """Args:
            model
            dataloader
            optimizer
            criterion
            kwargs:
                device
                epoch
        Return:
            (Accuracy, Loss) -  for history in view module."""
    model.train()
    running_loss = 0
    running_accuracy = 0
    if "epoch" not in kwargs:
        kwargs["epoch"] = '???'

    pbar = tqdm(dataloader,
                desc=f"Epoch: {kwargs['epoch']} --- Training... ",
                leave=False,
                unit="image")

    for inputs, labels in pbar:
        if "device" in kwargs:
            inputs = inputs.to(kwargs["device"])
            labels = labels.to(kwargs["device"])

        optimizer.zero_grad()

        predicted = model(inputs)
        loss = criterion(predicted, labels)

        running_loss += loss.item()
        running_accuracy += (labels == predicted.argmax(axis=1)).sum()

        loss.backward()

        optimizer.step()

    model.eval()

    return running_accuracy / len(dataloader.dataset), running_loss / len(dataloader)
