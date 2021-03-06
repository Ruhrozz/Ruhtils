"""Module for saving patterns and
further usage for augmentation tasks"""


from typing import Optional, Callable
import cv2
import albumentations as A
from albumentations.pytorch import transforms as T


def kbrodt_transforms(is_train: Optional[bool] = False,
                      p: Optional[float] = 0.5) -> Callable:
    """This function was taken from kbrodt GitHub
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
            A.Resize(256, 256),
            T.ToTensorV2(),
        ]
    else:
        augs = [
            A.Resize(256, 256),
            T.ToTensorV2(),
        ]

    augs_compose = A.Compose(augs)

    return augs_compose
