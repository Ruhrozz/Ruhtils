"""Module for saving patterns and
further usage for augmentation tasks"""


from typing import Optional, Callable
import cv2
import albumentations as A
from torchvision import transforms as T
from albumentations.pytorch import ToTensorV2


def kbrodt_transforms(is_train: Optional[bool] = False,
                      p: Optional[float] = 0.5) -> Callable:
    """Transforms were taken from kbrodt GitHub with adding normalization.
    Returns Albumentations.
    Args:
        is_train: Optional[bool]
            Return augs for valid or train data.
        p: Optional[float]
            Percentage of augmented data.
    Return:
        Composition of augmentations for train or valid data.
    """
    if is_train:
        augs = [
            A.Resize(256, 256),
            A.HorizontalFlip(p=p),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=p),
            A.OneOf(
               [
                   A.Blur(p=1),
                   A.GlassBlur(p=1),
                   A.GaussianBlur(p=1),
                   A.MedianBlur(p=1),
                   A.MotionBlur(p=1),
               ],
               p=p,
            ),
            A.RandomBrightnessContrast(p=p),
            A.OneOf(
               [
                   A.RandomGamma(p=1),  # works only for uint
                   A.ColorJitter(p=1),
                   A.RandomToneCurve(p=1),  # works only for uint
               ],
               p=p,
            ),
            A.OneOf(
               [
                   A.GaussNoise(p=1),
                   A.MultiplicativeNoise(p=1),
               ],
               p=p,
            ),
            A.OneOf(
                [
                    A.PiecewiseAffine(),
                    A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT),
                    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
                ],
                p=0.2,
            ),
            A.FancyPCA(p=0.2),
            A.RandomFog(p=0.2),
            A.RandomShadow(p=0.2),
            A.RandomSunFlare(src_radius=150, p=0.2),
            A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    else:
        augs = [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

    return A.Compose(augs)


def torch_transforms(is_train: Optional[bool] = False,
                     p: Optional[float] = 0.5) -> Callable:
    """Function returns a set of torchvision augmentations.
        Args:
            is_train: Optional[bool]
                Return augs for valid or train data.
            p: Optional[float]
                Percentage of augmented data.
        Return:
            Composition of augmentations for train or valid data.
        """
    if is_train:
        augs = [
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(p=p),
            T.RandomVerticalFlip(p=p),
            T.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
            T.ToTensor(),
            T.RandomErasing(p=p),
        ]
    else:
        augs = [
            T.Resize((256, 256)),
            T.ToTensor(),
        ]

    return T.Compose(augs)


def albu_transforms(is_train=False):
    """Example of albumentation augs"""
    if is_train:
        augs = [
            A.SmallestMaxSize(max_size=160),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=128, width=128),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    else:
        augs = [
            A.SmallestMaxSize(max_size=160),
            A.CenterCrop(height=128, width=128),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    return A.Compose(augs)
