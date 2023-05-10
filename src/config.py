import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# WandB


PROJECT = 'sex-estimation'
ENTITY = 'ivision-radiographs'
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
BATCH_SIZE = 128
PIN_MEMORY = True
LOAD_MODEL = False
CHECKPOINT_TO_BE_LOADED = './artifacts/model/my_checkpoint.pth.tar'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
WANDB_ON = False
MODEL_NAME = 'hiper_normal'
DATA_DIR = './data/binary/hiper_normal/'
# DATA_DIR = './data/binary/hiper_normal_reduced/'

train_transforms = A.Compose([
    A.Resize(width=224, height=224,),
    A.RandomCrop(width=224, height=224),
    A.Rotate(40),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])