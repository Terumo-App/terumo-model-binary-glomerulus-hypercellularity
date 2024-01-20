import PIL.Image
import albumentations as A
import albumentations.pytorch
import cv2

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
                 p: float = 1):
        super().__init__(always_apply, p)
        self.aspect_ratio: float | int = aspect_ratio
        self.pad_mode: str = pad_mode
        self.border_mode: cv2.BorderTypes
        self.border_value: float | int = 0
        if pad_mode == "zero_padding":
            self.border_mode = cv2.BORDER_CONSTANT
        elif pad_mode == "mirror_padding":
            self.border_mode = cv2.BORDER_REFLECT
        elif pad_mode == "replicate_padding":
            self.border_mode = cv2.BORDER_REPLICATE
        else:
            raise ValueError(f"Invalid padding mode for PadToAspectRatio transform: '{pad_mode}'")

    def apply(self, old_image: PIL.Image.Image, **params):
        curr_aspect_ratio = (old_image.size[0]) / (old_image.size[1])
        if curr_aspect_ratio == self.aspect_ratio:
            return old_image
        elif curr_aspect_ratio < self.aspect_ratio:
            # pad vertical dimension (height)
            target_h = int(self.aspect_ratio * old_image.size[1])
            transform = A.PadIfNeeded(min_height=target_h,
                                      min_width=0,
                                      border_mode=self.border_mode,
                                      value=self.border_value)
        else:
            # pad horizontal dimension (width)
            target_w = int(old_image.size[0] / self.aspect_ratio)
            transform = A.PadIfNeeded(min_height=0,
                                      min_width=target_w,
                                      border_mode=self.border_mode,
                                      value=self.border_value)
        return transform(old_image)["image"]


def get_resize_transform(resize_mode: str, img_size: tuple[int, int])\
        -> A.ImageOnlyTransform | A.DualTransform:
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
        return PadToAspectRatio(aspect_ratio=aspect_ratio,
                                pad_mode=resize_mode,
                                always_apply=True,
                                p=1)
    else:
        raise ValueError(f"Invalid resize mode: '{resize_mode}'")


def get_train_transform():
    albumentations_t = A.Compose([
        get_resize_transform(settings.image.RESIZE_MODE, settings.image.IMG_SIZE),
        A.Normalize(mean=settings.image.MEAN, std=settings.image.STD),
        A.HorizontalFlip(settings.augmentation.P_HORIZONTAL_FLIP),
        A.HorizontalFlip(settings.augmentation.P_VERTICAL_FLIP),
        A.Rotate(settings.augmentation.MAX_ROTATION_ANGLE, settings.augmentation.P_ROTATION),
        A.ColorJitter(
            brightness=settings.augmentation.BRIGHTNESS_FACTOR,
            contrast=settings.augmentation.CONTRAST_FACTOR,
            saturation=settings.augmentation.SATURATION_FACTOR,
            hue=settings.augmentation.HUE_FACTOR,
            p=settings.augmentation.P_COLOR_JITTER),
        A.GaussNoise(
            var_limit=settings.augmentation.GAUSS_NOISE_VAR_RANGE,
            mean=settings.augmentation.GAUSS_NOISE_MEAN,
            p=settings.augmentation.P_GAUSS_NOISE
        ),
        A.GaussianBlur(
            blur_limit=settings.augmentation.GAUSS_BLUR_LIMIT,
            p=settings.augmentation.P_GAUSS_BLUR
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
