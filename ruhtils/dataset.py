import cv2
from random import shuffle

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import DatasetFolder
from typing import Tuple, Optional, Callable, List, Any


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def default_loader(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


class ImageFolder(DatasetFolder):
    def __init__(self,
                 root: str,
                 use_albumentations: Optional[bool] = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root=root,
                         target_transform=target_transform,
                         extensions=IMG_EXTENSIONS,
                         loader=default_loader)

        if use_albumentations:
            self.transform = lambda image: transform(image=image)["image"]
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


class Dataset(VisionDataset):
    def __init__(
            self,
            root: str,
            samples: List[Tuple[str, int]],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            use_albumentations: bool = False,
    ) -> None:
        super().__init__(root=root)

        if use_albumentations and transform is not None:
            self.transform = lambda image: transform(image=image)["image"]
        else:
            self.transform = transform

        if use_albumentations and target_transform is not None:
            self.target_transform = lambda image: transform(image=image)["image"]
        else:
            self.target_transform = target_transform

        self.samples = samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = default_loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
