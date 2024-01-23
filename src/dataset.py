import cv2
import numpy as np

from torchvision.datasets import ImageFolder

class ImageFolderOverride(ImageFolder):
    def __init__(self,
                 root: str,
                 transform,
                 target_transform):
        def loader(img_path: str) -> np.ndarray:
            """
            Opens image path with OpenCV and convert the color channel disposition to the correct order
            (since OpenCV uses BGR and we usually use RGB (in numpy, torch, etc))
            Source: https://albumentations.ai/docs/examples/pytorch_classification/
            :param img_path: str
            :return: ndarray (image as numpy ndarray)
            """
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
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
