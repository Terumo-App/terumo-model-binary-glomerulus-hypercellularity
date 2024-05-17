from typing import Callable

import numpy as np
import torch

from PIL import Image

from torchvision.datasets import ImageFolder


class ImageFolderOverride(ImageFolder):
    def __init__(self,
                 root: str,
                 transform,
                 target_transform,
                 loader: Callable[[str], np.ndarray] | None = None):
        def base_loader(img_path: str) -> np.ndarray:
            """
            Opens image path with PIL (Python Imaging Library)

            Returns a channel-last image as a numpy ndarray
            (albumentations assumes a channel-last image and we use it for augmentation)
            :param img_path: str
            :return: ndarray (image as numpy ndarray, with channel-last dimension disposition)
            """
            image: Image = Image.open(img_path)
            image_arr: np.ndarray = np.asarray(image)  # channel-last, (H, W, C), see docs
            return image_arr

        if loader is None:
            super().__init__(root=root, transform=transform, target_transform=target_transform, loader=base_loader)
        else:
            super().__init__(root=root, transform=transform, target_transform=target_transform, loader=loader)

    def __getitem__(self, index: int):
        """
        Reimplements __getitem__ from torchvision.datasets.DatasetFolder (parent class of ImageFolder)
        to work with albumentations transforms, which need parameters to be passed via keywords
        and returns a dict (the transformed image being the value paired to the "image" key).

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
            and sample is a channel-last (C, H, W) image tensor.

            It keeps the type of the underlying transform pipeline (if it returns a numpy ndarray, this
            returns a ndarray, if it returns a torch Tensor, this returns a torch Tensor - else RuntimeError)
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(image=sample)["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)
        if isinstance(sample, np.ndarray):
            sample = np.moveaxis(sample, -1, 0)  # (C, H, W)
        elif not isinstance(sample, torch.Tensor):
            raise RuntimeError(f"Invalid type returned from augmentation pipeline: "
                               f"expected numpy.ndarray or torch.Tensor, got '{type(sample)}'")
        return sample, target

    def _get_class_name(self, index):
        return index
