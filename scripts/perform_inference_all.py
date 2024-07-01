import argparse
import datetime
import collections
import os
import re
import shutil

import numpy as np
import pandas as pd

from config import settings
from src.metrics import Metrics
from src.inference import inference


def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--test_dataset_root", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, default="min_loss")
    parser.add_argument("--best_model_dir", type=str, default="")
    parser.add_argument("--overwrite_best_model_dir", action="store_true", default=False)
    return parser


def find_last_checkpoint_dir(class_name: str, search_dir: str) -> str:
    all_checkpoint_dirs = os.listdir(search_dir)
    class_checkpoints = list(filter(lambda path: re.fullmatch(f"{class_name}.*", path) is not None,
                                    all_checkpoint_dirs))  # filter for name that matches class name
    class_checkpoints = list(filter(lambda path: os.path.isdir(os.path.join(search_dir, path)),
                                    class_checkpoints))  # filter for directories only
    try:
        print("Using checkpoint folder: ", sorted(class_checkpoints)[-1])
        return sorted(class_checkpoints)[-1]
    except IndexError:
        raise IndexError(f"No files match pattern '{class_name}.*' in directory {search_dir}")


def generate_checkpoint_files(class_name: str, search_dir: str, mode: str = "min_loss") -> list[str]:
    class_subdir = find_last_checkpoint_dir(class_name, search_dir)
    filtered_checkpoint_files = filter(lambda path: re.fullmatch(f".*{mode}.*", path) is not None,
                                       os.listdir(os.path.join(search_dir, class_subdir)))
    return list(map(lambda basepath: os.path.join(search_dir, class_subdir, basepath),
                    filtered_checkpoint_files))


def save_best_checkpoint(best_model_dir: str,
                         class_name: str,
                         checkpoint_files: list[str],
                         test_metrics: list[Metrics],
                         watch_metric: str = "fscore"):
    if len(checkpoint_files) != len(test_metrics):
        raise RuntimeWarning(f"List of checkpoint files and list of test metrics differ in size: "
                             f"{len(checkpoint_files)} and {len(test_metrics)}")
    idx: int = np.argmax([getattr(t, watch_metric) for t in test_metrics])
    checkpoint_basename: str = os.path.basename(checkpoint_files[idx])
    checkpoint_basename = "-".join([class_name, str(datetime.datetime.now()), checkpoint_basename])
    print(f"Best checkpoint for class {class_name} was {checkpoint_files[idx]}. Copying to {best_model_dir}...")
    shutil.copy(checkpoint_files[idx], os.path.join(best_model_dir, checkpoint_basename))


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    # clear best_model_dir
    if args.best_model_dir and args.overwrite_best_model_dir:
        print(f"Overwriting contents of directory {args.best_model_dir}")
        if os.path.isdir(args.best_model_dir):
            shutil.rmtree(args.best_model_dir)
        os.makedirs(args.best_model_dir, exist_ok=True)

    metrics_result = collections.defaultdict(list)
    # inference for each class
    for class_name in settings.data_processing.class_names:
        checkpoint_files = generate_checkpoint_files(class_name, search_dir=args.checkpoint_dir, mode=args.mode)
        test_losses: list[float] = []
        test_metrics: list[Metrics] = []
        
        metrics_result["class_name"].append(class_name)
        print(f" {class_name} binary classifier ".center(80, '-'))

        for checkpoint in checkpoint_files:
            if args.verbose:
                print(f"\nFold from checkpoint file '{checkpoint}'", end=2*'\n')
            loss, _, metrics_obj = inference(
                checkpoint_path=checkpoint,
                test_data_dir=os.path.join(args.test_dataset_root, f"binary_{class_name}"),
                verbose=args.verbose,
                device=args.device
            )
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
            metrics_result[m_name].append(mean)
            metrics_result[m_name + "_std"].append(std)


        if args.best_model_dir:  # != ""
            save_best_checkpoint(args.best_model_dir,
                                 class_name,
                                 checkpoint_files,
                                 test_metrics,
                                 watch_metric="fscore")

    df = pd.DataFrame(metrics_result)
    df.to_excel(f"./logs/metrics-{args.mode}-{str(datetime.datetime.now())}.xlsx", index=False)

if __name__ == '__main__':
    main()