import albumentations as A
import albumentations.pytorch
import cv2
import numpy as np

from typing import Sequence

from config import settings


class PadToAspectRatio(A.ImageOnlyTransform):
    """
    Custom albumentation-compatible transform
    (see https://stackoverflow.com/questions/75903878/use-custom-transformer-for-albumentations).

    Pads the smaller image dimension such that the aspect ratio is a given constant
    (up to the precision error inherent to using integer sizes).
    """
    def __init__(self,
                 aspect_ratio: float | int = 1,
                 pad_mode: str = "zero_padding",
                 always_apply: bool = False,
                 p: float = 1,
                 border_value: float | Sequence[float] = 0):
        super().__init__(always_apply, p)
        self.aspect_ratio: float | int = aspect_ratio
        self.pad_mode: str = pad_mode
        self.border_mode: cv2.BorderTypes
        self.border_value: float | Sequence[float] = 0
        if pad_mode == "zero_padding":
            self.border_mode = cv2.BORDER_CONSTANT
            self.border_value = border_value
        elif pad_mode == "mirror_padding":
            self.border_mode = cv2.BORDER_REFLECT
        elif pad_mode == "replicate_padding":
            self.border_mode = cv2.BORDER_REPLICATE
        else:
            raise ValueError(f"Invalid padding mode for PadToAspectRatio transform: '{pad_mode}'")

    def apply(self, old_image: np.ndarray, **params) -> np.ndarray:
        """
        Let R_0 = H / W be the current aspect ratio and R the target aspect ratio
        If R_0 < R:
            resizes H to H_2 := floor(W * R)
        else if R_0 > R:
            resizes W to W_2 := floor(H / R)

        :param old_image: a (H, W, C) numpy array
        :param params:
        :return: a new_image of size (H_2, W_2, C)
            where H_2 = max(H, floor(W * R))
            and W_2 = max(W, floor(H / R))
        """
        h, w = old_image.shape[0], old_image.shape[1]
        min_h = max(h, int(w * self.aspect_ratio))
        min_w = max(w, int(h / self.aspect_ratio))
        transform = A.PadIfNeeded(min_height=min_h,
                                  min_width=min_w,
                                  border_mode=self.border_mode,
                                  value=self.border_value)
        return transform(image=old_image)["image"]


def get_resize_transform(resize_mode: str, img_size: tuple[int, int])\
        -> A.ImageOnlyTransform | A.DualTransform | A.Compose:
    if resize_mode == "interpolation":
        return A.Resize(height=img_size[0], width=img_size[1], always_apply=True, p=1)
    elif resize_mode == "random_crop":
        return A.RandomResizedCrop(height=img_size[0], width=img_size[1], always_apply=True, p=1)
    elif resize_mode == "strict_zero_padding":
        # this is not recommended, since it will pad to be EXACTLY this size
        # and not just the same aspect ratio
        return A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1])
    elif (resize_mode == "zero_padding") or (resize_mode == "mirror_padding") \
        or (resize_mode == "replicate_padding"):
        aspect_ratio = img_size[0] / img_size[1]
        pad_t = PadToAspectRatio(aspect_ratio=aspect_ratio,
                                pad_mode=resize_mode,
                                always_apply=True,
                                p=1)
        resize_no_distortion_t = A.Resize(height=img_size[0],
                                          width=img_size[1],
                                          always_apply=True,
                                          p=1)
        return A.Compose([pad_t, resize_no_distortion_t])
    else:
        raise ValueError(f"Invalid resize mode: '{resize_mode}'")


def get_train_transform():
    albumentations_t = A.Compose([
        get_resize_transform(settings.image.RESIZE_MODE, settings.image.IMG_SIZE),
        A.Normalize(mean=settings.image.MEAN, std=settings.image.STD),
        A.HorizontalFlip(settings.augmentation.P_HORIZONTAL_FLIP),
        A.VerticalFlip(settings.augmentation.P_VERTICAL_FLIP),
        A.Rotate(limit=settings.augmentation.MAX_ROTATION_ANGLE,
                 p=settings.augmentation.P_ROTATION),
        A.ColorJitter(
            brightness=settings.augmentation.COLOR.BRIGHTNESS_FACTOR,
            contrast=settings.augmentation.COLOR.CONTRAST_FACTOR,
            saturation=settings.augmentation.COLOR.SATURATION_FACTOR,
            hue=settings.augmentation.COLOR.HUE_FACTOR,
            p=settings.augmentation.COLOR.P_COLOR_JITTER),
        A.GaussNoise(
            var_limit=settings.augmentation.NOISE.GAUSS_NOISE_VAR_RANGE,
            mean=settings.augmentation.NOISE.GAUSS_NOISE_MEAN,
            p=settings.augmentation.NOISE.P_GAUSS_NOISE
        ),
        A.GaussianBlur(
            blur_limit=settings.augmentation.NOISE.GAUSS_BLUR_LIMIT,
            p=settings.augmentation.NOISE.P_GAUSS_BLUR
        ),
        A.CoarseDropout(
            max_holes=settings.augmentation.COARSE_DROPOUT.MAX_HOLES,
            max_height=settings.augmentation.COARSE_DROPOUT.MAX_H,
            max_width=settings.augmentation.COARSE_DROPOUT.MAX_W,
            min_holes=settings.augmentation.COARSE_DROPOUT.MIN_HOLES,
            min_height=settings.augmentation.COARSE_DROPOUT.MIN_H,
            min_width=settings.augmentation.COARSE_DROPOUT.MIN_W,
            fill_value=0,
            mask_fill_value=0,
            p=settings.augmentation.COARSE_DROPOUT.P_COARSE_DROPOUT),
        A.OneOf(
            [
                A.OpticalDistortion(p=settings.augmentation.DISTORTION.P_OPTICAL_DISTORTION),
                A.GridDistortion(p=settings.augmentation.DISTORTION.P_GRID_DISTORTION),
                A.PiecewiseAffine(p=settings.augmentation.DISTORTION.P_PIECEWISE_AFFINE),
            ],
            p=settings.augmentation.DISTORTION.P_DISTORTION
        ),
        A.ShiftScaleRotate(
            shift_limit=settings.augmentation.SHIFT.SHIFT_LIMIT,
            scale_limit=settings.augmentation.SHIFT.SCALE_LIMIT,
            rotate_limit=settings.augmentation.SHIFT.ROTATE_LIMIT,
            interpolation=cv2.INTER_LINEAR,
            border_mode=0,
            value=(0, 0, 0),
            p=settings.augmentation.SHIFT.P_SHIFT
        ),
        A.pytorch.ToTensorV2(),
    ])
    return albumentations_t


def get_test_transform():
    albumentations_t = A.Compose([
        get_resize_transform(settings.image.RESIZE_MODE, settings.image.IMG_SIZE),
        A.Normalize(mean=settings.image.MEAN, std=settings.image.STD),
        A.pytorch.ToTensorV2(),
    ])
    return albumentations_t
