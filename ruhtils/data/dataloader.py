from torch.utils.data import DataLoader


# TODO: get distributed dataloader function
def get_dataloader(dataset, device, **dataloader_kwargs):

    dataloader = DataLoader(dataset,
                            pin_memory=True if device == "GPU" else False,
                            **dataloader_kwargs)

    return dataloader
