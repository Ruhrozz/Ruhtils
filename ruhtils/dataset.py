import cv2
from random import shuffle
from torchvision.datasets.folder import DatasetFolder
from typing import Any, Tuple, Optional, Callable, List


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def loader(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


class ImageFolder(DatasetFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root=root,
                         transform=transform,
                         target_transform=target_transform,
                         extensions=IMG_EXTENSIONS,
                         loader=loader)

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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(image=sample)["image"]

        if self.target_transform is not None:
            target = self.target_transform(image=target)["image"]

        return sample, target
