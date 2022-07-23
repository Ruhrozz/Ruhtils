import os
import cv2
import numpy as np
from random import shuffle
from torchvision.datasets import VisionDataset
from typing import Any, Dict, List, Tuple, Optional, Callable, Union


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Find the class folders in a dataset structured as follows::
        directory/
        ├── class_x
        │   ├── xxx.ext
        │   ├── xxy.ext
        │   └── ...
        │       └── xxz.ext
        └── class_y
            ├── 123.ext
            ├── nsdf3.ext
            └── ...
            └── asd932_.ext

    Args:
        directory: Root directory path
    Raises:
        FileNotFoundError: If ``dir`` has no class folders.
    Returns:
        (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
    """

    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(directory: str, extensions: Union[str, Tuple[str, ...]], ) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
        Raises:
            FileNotFoundError: In case no valid file was found for any class.
        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """

    directory = os.path.expanduser(directory)
    _, class_to_idx = find_classes(directory)

    def is_valid_file(x: str) -> bool:
        return has_file_allowed_extension(x, extensions)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader.
        This default directory structure can be customized by overriding the
        :meth:`find_classes` method.
        Args:
            root (string): Root directory path.'
            samples (List[Tuple[str, int]]): Optional list of ready-to-read
                files and class indexes.
            extensions (tuple[string]): A list of allowed extensions.
            transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            target_transform (callable, optional): A function/transform that takes
                in the target and transforms it.
            to_rgb (bool): whether convert cv2 BGR to RGB
         Attributes:
            classes (list): List of the class names sorted alphabetically.
            class_to_idx (dict): Dict with items (class_name, class_index).
            samples (list): List of (sample path, class_index) tuples
            targets (list): The class_index value for each image in the dataset
        """

    def __init__(
            self,
            root: str,
            samples: Optional[List[Tuple[str, int]]] = None,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            to_rgb: Optional[bool] = False
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = find_classes(self.root)

        if samples is None:
            samples = make_dataset(self.root, extensions)

        self.extensions = IMG_EXTENSIONS

        if extensions is not None:
            self.extensions = extensions

        self.to_rgb = to_rgb

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = cv2.imread(path)

        if self.to_rgb:
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


def make_train_valid(root: str,
                     sample: float = 0.2,
                     extensions: Optional[Tuple[str]] = None,
                     transform: Optional[Tuple[Callable, Callable]] = (None, None),
                     target_transform: Optional[Tuple[Callable, Callable]] = (None, None),
                     to_rgb: Optional[bool] = False) -> Tuple[DatasetFolder, DatasetFolder]:

    kwargs_train = kwargs_valid = {"root": root,
                                   "extensions": extensions,
                                   "to_rgb": to_rgb}
    if extensions is None:
        dataset = make_dataset(root, IMG_EXTENSIONS)
    else:
        dataset = make_dataset(root, extensions)

    shuffle(dataset)
    threshold = int(len(dataset) * sample)
    train_set = dataset[:threshold]
    valid_set = dataset[threshold:]

    assert len(train_set) + len(valid_set) == len(dataset)

    kwargs_train.update({"samples": train_set, "transform": transform[0], "target_transform": target_transform[0]})
    kwargs_valid.update({"samples": valid_set, "transform": transform[1], "target_transform": target_transform[1]})

    return DatasetFolder(**kwargs_train), DatasetFolder(**kwargs_valid)
