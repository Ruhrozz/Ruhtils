"""Model training function"""


from typing import Tuple
import torch
from tqdm import tqdm


def _kwargs_check(kwargs):
    if "epoch" not in kwargs:
        kwargs["epoch"] = '???'
    if "is_train" not in kwargs:
        kwargs["is_train"] = False
    if kwargs['is_train']:
        kwargs['desc'] = "Training"
    else:
        kwargs['desc'] = "Validating"
    return kwargs


def _valid_fn(model, criterion, inputs, labels):
    with torch.no_grad():
        predicted = model(inputs)
        loss = criterion(predicted, labels)
        return loss.item(), (labels == predicted.argmax(axis=1)).sum()


def _train_fn(model, optimizer, criterion, inputs, labels):
    optimizer.zero_grad()
    predicted = model(inputs)
    loss = criterion(predicted, labels)
    loss.backward()
    optimizer.step()
    return loss.item(), (labels == predicted.argmax(axis=1)).sum()


def trainval(model, dataloader, criterion, optimizer=None, **kwargs) -> Tuple[float, float]:
    """Args:
            model
            dataloader
            criterion
            optimizer: Optional if valid
            kwargs:
                device: torch.device
                epoch: int
                is_train: bool
        Return:
            (Accuracy, Loss) -  for history in view module."""
    running_accuracy = 0
    running_loss = 0
    _kwargs_check(kwargs)

    pbar = tqdm(dataloader,
                leave=False,
                unit="image",
                desc=f"Epoch: {kwargs['epoch']} --- {kwargs['desc']}... ")
    model.train()
    if not kwargs['is_train']:
        model.eval()

    for inputs, labels in pbar:
        if "device" in kwargs:
            inputs = inputs.to(kwargs["device"])
            labels = labels.to(kwargs["device"])
        if kwargs['is_train']:
            loss, acc = _train_fn(model, optimizer, criterion, inputs, labels)
        else:
            loss, acc = _valid_fn(model, criterion, inputs, labels)
        running_loss += loss
        running_accuracy += acc

    return running_accuracy / len(dataloader.dataset), running_loss / len(dataloader)
