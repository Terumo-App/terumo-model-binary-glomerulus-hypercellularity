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
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--mcd", "--num_samples_mc_dropout", 
                        help="Number of times to run inference for a single dataset sample" 
                        "(in the way given the Monte Carlo dropout/MCD method). Default = 1 (no MCD)",
                        type=int, default=1)
    parser.add_argument("--dropout", help="Probability of dropout on inference (between 0.0 and 1.0)."
                        "Assumes Monte Carlo dropout method. Default = 0 (no dropout)",
                        type=float, default=0)
    return parser


def setup_model(checkpoint_path: str, device: str, p_dropout: float, **opt_params) -> tuple[Net, torch.optim.Adam]:
    model = Net(net_version="b0", num_classes=2, p_dropout=p_dropout)
    opt = torch.optim.Adam(model.parameters(), **opt_params)
    load_checkpoint(checkpoint_path, model, opt)
    model = model.to(device)
    return model, opt


def log_inference(loss: float, metrics: Metrics, total_samples: int, var_matrices: torch.Tensor | None) -> None:
    print(f" Test epoch ".center(80, '-'))
    print(f"test/loss: {loss} (total on dataset) -> Total={total_samples} test images")
    print(f"test/loss: {loss / total_samples} (average per image)")
    for m_name in metrics.__dict__:
        if isinstance(getattr(metrics, m_name), float):
            print(f"test/{m_name} = {100 * getattr(metrics, m_name):.4f}%")
    if var_matrices is not None:
        print(f"Average variance matrix over dataset:\n{torch.mean(var_matrices, dim=0)}")


def inference(checkpoint_path: str,
              config_file_path: str,
              test_data_dir: str,
              device: str = "cpu",
              verbose: bool = True,
              p_dropout: float = 0,
              num_samples_mc_dropout: int = 1) -> tuple[float, int, Metrics]:
    run_params = load_training_parameters(config_file_path)
    model, opt = setup_model(checkpoint_path, device, lr=run_params["learning_rate"], p_dropout=p_dropout)

    def in_the_wild_loader(path: str) -> np.ndarray:
        rgb_image = Image.open(path).convert("RGB")
        return np.asarray(rgb_image)

    test_dataset = ImageFolderOverride(root=test_data_dir,
                                       transform=get_test_transform(),
                                       target_transform=lambda index: index,
                                       loader=in_the_wild_loader)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=run_params["batch_size"],
                                                  shuffle=False,
                                                  num_workers=0)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    if verbose:
        print(f"Running on device='{device}'")
        print(f"Testing on {len(test_dataset)} images")
    if num_samples_mc_dropout > 1:
        test_loss, test_correct_cnt, test_metrics, var_matrices = valid_epoch(loader=test_dataloader,
                                                                              model=model,
                                                                              loss_fn=loss_fn,
                                                                              device=device,
                                                                              monte_carlo_runs=num_samples_mc_dropout)
    else:
        test_loss, test_correct_cnt, test_metrics = valid_epoch(loader=test_dataloader,
                                                                model=model,
                                                                loss_fn=loss_fn,
                                                                device=device,
                                                                monte_carlo_runs=num_samples_mc_dropout)
        var_matrices = None

    if verbose:
        log_inference(test_loss, test_metrics, len(test_dataset), var_matrices)
    return test_loss, test_correct_cnt, test_metrics


if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()
    inference(args.checkpoint,
              args.config_file,
              args.test_dataset,
              args.device,
              args.verbose,
              args.dropout,
              args.mcd)