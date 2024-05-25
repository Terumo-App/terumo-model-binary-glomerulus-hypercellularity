import argparse
import numpy as np
import torch.utils.data
import torchvision.datasets
import albumentations as A

from PIL import Image

from src.dataset import ImageFolderOverride
from src.metrics import Metrics
from src.model import Net
from src.utils import valid_epoch
from src.utils import load_checkpoint, load_training_parameters
from src.transforms import get_test_transform


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def setup_model(checkpoint_path: str, device: str, **opt_params) -> tuple[Net, torch.optim.Adam]:
    model = Net(net_version="b0", num_classes=2)
    opt = torch.optim.Adam(model.parameters(), **opt_params)
    load_checkpoint(checkpoint_path, model, opt)
    model = model.to(device)
    return model


def log_inference(loss: float, metrics: Metrics, total_samples: int) -> None:
    print(f" Test epoch ".center(80, '-'))
    print(f"test/loss: {loss} (total on dataset) -> Total={total_samples} test images")
    print(f"test/loss: {loss / total_samples} (average per image)")
    for m_name in metrics.__dict__:
        if isinstance(getattr(metrics, m_name), float):
            print(f"test/{m_name} = {100 * getattr(metrics, m_name):.4f}%")


def inference(checkpoint_path: str,
              test_data_dir: str,
              device: str = "cpu",
              verbose: bool = True) -> tuple[float, int, Metrics]:
    model = setup_model(checkpoint_path, device)

    def in_the_wild_loader(path: str) -> np.ndarray:
        rgb_image = Image.open(path).convert("RGB")
        return np.asarray(rgb_image)

    test_dataset = ImageFolderOverride(root=test_data_dir,
                                       transform=get_test_transform(),
                                       target_transform=lambda index: index,
                                       loader=in_the_wild_loader)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=100,
                                                  shuffle=False,
                                                  num_workers=0)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    if verbose:
        print(f"Running on device='{device}'")
        print(f"Testing on {len(test_dataset)} images")
    test_loss, test_correct_cnt, test_metrics = valid_epoch(loader=test_dataloader,
                                                            model=model,
                                                            loss_fn=loss_fn,
                                                            device=device)
    if verbose:
        log_inference(test_loss, test_metrics, len(test_dataset))
    return test_loss, test_correct_cnt, test_metrics


if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()
    inference(args.checkpoint, args.test_dataset, args.device, args.verbose)