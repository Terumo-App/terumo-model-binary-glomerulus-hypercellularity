import numpy as np

from PIL import Image

from torchvision.datasets import ImageFolder

class ImageFolderOverride(ImageFolder):
    def __init__(self,
                 root: str,
                 transform,
                 target_transform):
        def loader(img_path: str) -> np.ndarray:
            """
            Opens image path with PIL (Python Imaging Library)

            Returns a channel-first image as a numpy ndarray (since pytorch uses channel-first images)
            :param img_path: str
            :return: ndarray (image as numpy ndarray, with channel-first dimension disposition)
            """
            image: Image = Image.open(img_path)
            image_arr: np.ndarray = np.asarray(image)  # channel-last, (H, W, C), see docs
            return np.moveaxis(image_arr, -1, 0)  # (C, H, W)
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
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(image=sample)["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _get_class_name(self, index):
        return index
