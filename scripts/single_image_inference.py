"""
This scripts is an example script to illustrate model inference
    from the weights retrieved from the best models for each lesion.

Usage:
    python3 inference.py [path-to-image] [path-to-models-directory]

IMPORTANT NOTE(1): the second argument is a path to the directory which contains the model weights / checkpoint (.pth.tar file) for each binary classifier.

IMPORTANTE NOTE(2): the classifier returns two scalars (logits, unnormalized). After applying the softmax, the FIRST will be the score/probability of being POSITIVE for that class, and the SECOND will be the complementary probability of being NEGATIVE.

The output will be a series of 6 lines of the form "Lesion: score for that lesion" where the score is a floating point number between 0.0 and 1.0.
"""
import numpy as np
import torch
import albumentations as A
import cv2
import os
import sys

from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path
from torch import nn
from efficientnet_pytorch import EfficientNet

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        # overwrite MLP / decoder da EfficientNet B0
        self.backbone._fc = nn.Sequential(
            nn.Linear(1280, 2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def load_and_preprocess(image_path: str) -> torch.Tensor:
    """
    Loads image from filepath to a numpy ndarray.
    Then takes ndarray and apply transforms (padding, resizing, normalization) and transforms to tensor

    NOTE: numpy images (from PIL) are channel-last (form (H, W, 3) in case of RGB / 3 channels)

    outputs a channel-last torch (3, 224, 224) tensor, already normalized to mean=[0.5,0.5,0.5] and std=[0.5,0.5,0.5]
    """
    TARGET_H = 224
    TARGET_W = 224
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]

    numpy_img = np.asarray(Image.open(image_path).convert("RGB"))

    h, w = numpy_img.shape[0], numpy_img.shape[1]
    zero_padding_to_square = A.PadIfNeeded(min_height=max(h, w), min_width=max(h, w), border_mode=cv2.BORDER_CONSTANT, value=0)
    resize = A.Resize(height=TARGET_H, width=TARGET_W, always_apply=True)
    normalize = A.Normalize(mean=MEAN, std=STD, always_apply=True)
    to_tensor = ToTensorV2(always_apply=True)

    pipeline = A.Compose([zero_padding_to_square, resize, normalize, to_tensor])
    # NOTE albumentations transforms must receive the image as
    # a keyword argument called image
    # or we get "KeyError: 'You have to pass data to augmentations as named arguments, for example: aug(image=image)'"
    return pipeline(image=numpy_img)["image"]

def evaluate_scores(img: torch.Tensor, models_directory: str) -> dict[str, float]:
    class_names = [
        "Crescent",
        "Hypercellularity",
        "Podocytopathy",
        "Normal",
        "Sclerosis",
        "Membranous"
    ]
    def find_checkpoint(paths: list[str], class_name: str):
        for p in paths:
            if p.startswith(class_name):
                return p

    for cls in class_names:
        binary_classifier = Net()
        checkpoint_path = Path(models_directory) / find_checkpoint(os.listdir(models_directory), cls)
        model_weights = torch.load(checkpoint_path)["state_dict"]
        binary_classifier.load_state_dict(model_weights)
        
        # puts in evaluation mode, turning off dropout
        # very important line
        binary_classifier.eval()

        # efficient net expects a 4D input (batch_size, channels, H, W)
        logit = binary_classifier(img.unsqueeze(dim=0))
        scores = torch.softmax(logit, dim=-1).squeeze(dim=0)
        # the first is the probability of being positive
        print(f"Score/probability of being class {cls}: {scores[0]}")
        

if __name__ == "__main__":
    image_path = sys.argv[1]
    models_directory = sys.argv[2]
    processed_img = load_and_preprocess(image_path)
    evaluate_scores(processed_img, models_directory=models_directory)