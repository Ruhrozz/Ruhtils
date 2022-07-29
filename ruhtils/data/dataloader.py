"""
This module contains functions of creating
dataloaders for different tasks and devices.
"""


from torch.utils.data import DataLoader


# TODO: get distributed dataloader function
def get_dataloader(dataset, device: str, **dataloader_kwargs):
    """ This function makes dataloader from dataset.
    Args:
        dataset:
            Torch dataset
        device:
            Any of ["GPU", "CPU", "TPU"]
        **dataloader_kwargs:
            Keyword arguments that will be passed directly
            in torch.utils.data.Dataloader function.
    Return:
        dataloader

    """

    dataloader = DataLoader(dataset,
                            pin_memory=(device == "GPU"),
                            **dataloader_kwargs)

    return dataloader
