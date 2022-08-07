"""
This module contains functions and classes
for getting and creating datasets.
"""


from random import shuffle
from typing import Tuple, Optional, Callable, List, Any

import cv2
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets import ImageFolder as ImageFolder_torch


def cv_loader(image_path: str) -> np.ndarray:
    """ Image loader for class ImageFolder using cv2.
    Args:
        image_path: str
            Path like "C:/Pictures/image1.png"
    Return:
        np.ndarray
            Image [Channels, Height, Width]
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def pil_loader(path: str) -> Image.Image:
    """ Image loader for class ImageFolder using PIL.
        Args:
            path: str
                Path like "C:/Pictures/image1.png"
        Return:
            np.ndarray
                Image [Channels, Height, Width]
        """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class Dataset(VisionDataset):
    """This class makes a dataset from image paths and labels.
    Args:
        samples: List[Tuple[str, int]]
            Dataset samples [("path":str, class:int), ...]
        transform: Optional[Callable]
            Torch or Albumentations transform compose.
        target_transform: Optional[Callable]
            This function will be applied to dataset's labels.
        use_albumentations: Optional[bool]
            Whether to use albumentations transform semantic.
    """
    def __init__(
            self,
            samples: List[Tuple[str, int]],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            use_albumentations: bool = False,
    ) -> None:
        super().__init__(root="no root",
                         target_transform=target_transform)

        if use_albumentations and transform is not None:
            self.transform = lambda image: transform(image=image)["image"]
            self.loader = cv_loader
        else:
            self.transform = transform
            self.loader = pil_loader

        self.samples = samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class ImageFolder(ImageFolder_torch):
    """This class is the same thing as ImageFolder class from torchvision.
    see also:
    https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py
    Args:
        root: str
            Where pictures will be taken from.
        transform: Optional[Callable]
            Torch or Albumentations transform compose.
        target_transform: Optional[Callable]
            This function will be applied to dataset's labels.
        use_albumentations: Optional[bool]
            Whether to use albumentations transform semantic.
    """
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_albumentations: Optional[bool] = False) -> None:
        super().__init__(root=root,
                         target_transform=target_transform)

        if use_albumentations and transform is not None:
            self.transform = lambda image: transform(image=np.array(image))["image"]
        else:
            self.transform = transform

    def take_valid(self, sample: float = 0.2) -> List[Tuple[str, int]]:
        """Splits sample to 2 disjoint sets with given validation percentage
        Args:
            sample: percentage of the validation sample
        Returns:
            List[Tuple[str, int]]: valid samples of a form (path_to_sample, class)
        """
        shuffle(self.samples)

        threshold = int(len(self.samples) * sample)

        valid = self.samples[:threshold]
        self.samples = self.samples[threshold:]
        self.targets = [s[1] for s in self.samples]

        return valid
