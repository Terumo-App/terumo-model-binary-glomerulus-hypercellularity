from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import settings

class ImageDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize(settings.image.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=settings.image.mean, std=settings.image.std)
        ])
        self.dataset = ImageFolder(self.data_dir, transform=self.transform, target_transform=self._get_class_name)
     
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataloader)

    def _get_class_name(self, index):
        return index
