import argparse

import os
import re

import numpy as np

from config import settings
from src.metrics import Metrics
from src.inference import inference

CONFIG_FILES_MAPPING = {
    "Hypercellularity": "config/01_hipercel.yaml",
    "Membranous": "config/02_membra.yaml",
    "Sclerosis": "config/03_sclerosis.yaml",
    "Normal": "config/04_normal.yaml",
    "Podocytopathy": "config/05_podoc.yaml",
    "Crescent": "config/06_cresce.yaml",
}


def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--test_dataset_root", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, default="min_loss")
    return parser

def find_last_checkpoint_dir(class_name: str, search_dir: str) -> str:
    all_checkpoint_dirs = os.listdir(search_dir)
    class_checkpoints = list(filter(lambda path: re.fullmatch(f"{class_name}.*", path) is not None,
                                    all_checkpoint_dirs))  # filter for name that matches class name
    class_checkpoints = list(filter(lambda path: os.path.isdir(os.path.join(search_dir, path)),
                                    class_checkpoints))  # filter for directories only
    try:
        return sorted(class_checkpoints)[-1]
    except IndexError:
        raise IndexError(f"No files match pattern '{class_name}.*' in directory {search_dir}")


def generate_checkpoint_files(class_name: str, search_dir: str, mode: str = "min_loss") -> list[str]:
    class_subdir = find_last_checkpoint_dir(class_name, search_dir)
    filtered_checkpoint_files = filter(lambda path: re.fullmatch(f".*{mode}.*", path) is not None,
                                       os.listdir(os.path.join(search_dir, class_subdir)))
    return list(map(lambda basepath: os.path.join(search_dir, class_subdir, basepath),
                    filtered_checkpoint_files))


def main():
    parser = setup_argparse()
    args = parser.parse_args()
    for class_name in settings.data_processing.class_names:
        config_file = CONFIG_FILES_MAPPING[class_name]
        checkpoint_files = generate_checkpoint_files(class_name, search_dir=args.checkpoint_dir, mode=args.mode)
        test_losses: list[float] = []
        test_metrics: list[Metrics] = []

        print(f" {class_name} binary classifier ".center(80, '-'))
        for checkpoint in checkpoint_files:
            if args.verbose:
                print(f"Fold from checkpoint file '{checkpoint}'", end=2*'\n')
            loss, _, metrics_obj = inference(checkpoint_path=checkpoint,
                                             config_file_path=config_file,
                                             test_data_dir=os.path.join(args.test_dataset_root, f"binary_{class_name}"),
                                             verbose=args.verbose,
                                             device=args.device)
            if args.verbose:
                print()
            test_losses.append(loss)
            test_metrics.append(metrics_obj)

        metrics_names = [key for key in test_metrics[0].__dict__\
                         if isinstance(getattr(test_metrics[0], key), float)]
        print(f"test/loss (mean): {np.mean(test_losses)}")
        print(f"test/loss (std): {np.std(test_losses)}")
        for m_name in metrics_names:
            mean = np.mean([getattr(t, m_name) for t in test_metrics])
            std = np.std([getattr(t, m_name) for t in test_metrics])
            print(f"test/{m_name} (mean): {mean}")
            print(f"test/{m_name} (std): {std}")


if __name__ == '__main__':
    main()