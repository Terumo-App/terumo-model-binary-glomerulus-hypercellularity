import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import wandb

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from src.model import Net
from src.utils import (
     valid_epoch, 
     load_checkpoint, 
     save_checkpoint, 
     make_prediction, 
     get_balanced_dataset_sampler,
     train_epoch,
     create_timestamp_folder, 
     initialize_wandb, 
     set_gpu_mode, 
     measure_execution_time,
     load_training_parameters,
     wandb_log_final_result
)
from src.transforms import get_train_transform, get_test_transform
from src.dataset import ImageFolderOverride
from src.options import BaseOptions
from src.stopper import EarlyStopper
from config import settings

# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
# But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.
torch.backends.cudnn.benchmark = True
opt = BaseOptions().parse()
PARAMS = load_training_parameters(opt.config_file)
wandb.login(key=PARAMS['wandb_key'])
CHECKPOINT_ARTIFACT_MIN_LOSS: wandb.Artifact | None = wandb.Artifact(name="checkpoint_min_loss", type="model")\
    if (PARAMS["wandb_on"] and PARAMS["wandb_save_checkpoint"]) else None
CHECKPOINT_ARTIFACT_MAX_ACC: wandb.Artifact | None = wandb.Artifact(name="checkpoint_max_acc", type="model") \
    if (PARAMS["wandb_on"] and PARAMS["wandb_save_checkpoint"]) else None


def setup_early_stopper(params) -> EarlyStopper:
    patience = PARAMS["patience"] if "patience" in PARAMS else float("inf")
    delta = PARAMS["delta"] if "delta" in PARAMS else 0
    mode = PARAMS["mode"] if "mode" in PARAMS else "min"
    threshold_mode = PARAMS["threshold_mode"] if "threshold_mode" in PARAMS else "min"
    threshold = PARAMS["threshold"] if "threshold" in PARAMS else None

    early_stopper = EarlyStopper(
        patience=patience,
        mode=mode,
        delta=delta,
        threshold=threshold,
        threshold_mode=threshold_mode
    )
    return early_stopper


def main():
    if settings.config.DEVICE == 'cuda' and not torch.cuda.is_available():
        raise ValueError("DEVICE is set to cuda but cuda is not available")

    print(f'Device is {settings.config.DEVICE}')
    
    artifact_folder = create_timestamp_folder(PARAMS['model_name'])
    skfold = StratifiedKFold(n_splits=PARAMS['n_folds'], shuffle=True)
    full_dataset_train_mode = ImageFolderOverride(root=PARAMS['data_dir'],
                                                  transform=get_train_transform(),
                                                  target_transform=lambda index: index)
    full_dataset_val_mode = ImageFolderOverride(root=PARAMS['data_dir'],
                                                transform=get_test_transform(),
                                                target_transform=lambda index: index)
    binary_labels = [sample[1] for sample in full_dataset_train_mode.samples]
    early_stopper = setup_early_stopper(PARAMS)

    for fold, (train_ids, val_ids) in enumerate(skfold.split(full_dataset_train_mode, binary_labels)):
        # reset early stopper
        early_stopper.reset()

        initialize_wandb(PARAMS, fold+1, artifact_folder,
                         train_dataset=len(train_ids),
                         val_dataset=len(val_ids))
        train_subset = Subset(full_dataset_train_mode, train_ids)
        sampler = get_balanced_dataset_sampler(full_dataset_train_mode, train_ids, train_subset)
        train_loader = DataLoader(train_subset,
                                  batch_size=PARAMS['batch_size'],
                                  sampler=sampler,
                                  num_workers=PARAMS['num_workers'])
        val_subset = Subset(full_dataset_val_mode, val_ids)
        val_loader = DataLoader(val_subset,
                                batch_size=PARAMS['batch_size'],
                                num_workers=PARAMS['num_workers'],
                                shuffle=True)
        print(f'Fold  {fold +1}')

        loss_fn = nn.CrossEntropyLoss()
        model = Net(net_version=settings.model.net_version, num_classes=2, freeze=PARAMS["freeze"]).to(settings.config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=PARAMS['learning_rate'])
        scaler = torch.cuda.amp.GradScaler()
        set_gpu_mode(model)

        if PARAMS['load_model']:
            load_checkpoint(torch.load(PARAMS['checkpoint_to_be_loaded']), model, optimizer)

        max_val_accuracy, min_val_loss = 0, sys.maxsize
        for epoch in range(PARAMS['num_epochs']):
            train_results = train_epoch(train_loader, model, optimizer, loss_fn, scaler, settings.config.DEVICE)
            val_results = valid_epoch(val_loader, model, loss_fn, settings.config.DEVICE)

            train_loss, train_correct = train_results
            val_loss, val_correct, val_metrics = val_results

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            val_loss = val_loss / len(val_loader.sampler)
            val_acc = val_correct / len(val_loader.sampler) * 100
  

            print(f"Epoch:{epoch + 1}/{PARAMS['num_epochs']} AVG Training Loss:{train_loss:.3f}, AVG Test Loss:{val_loss:.3f}, AVG Training Acc {train_acc:.2f}%, AVG Test Acc {val_acc:.2f}% , Test F1 Score {val_metrics.fscore*100:.2f}%")
            
            if PARAMS['wandb_on']:
                wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_fscore': val_metrics.fscore*100,
                })

            if max_val_accuracy < val_acc:
                max_val_accuracy = val_acc
                save_checkpoint(model, optimizer, artifact_folder, 'max_acc', fold,
                                log_to_wandb=PARAMS["wandb_on"] and PARAMS["wandb_save_checkpoint"],
                                checkpoint_artifact=CHECKPOINT_ARTIFACT_MAX_ACC)

            if min_val_loss > val_loss:
                min_val_loss = val_loss
                save_checkpoint(model, optimizer, artifact_folder, 'min_loss', fold,
                                log_to_wandb = PARAMS["wandb_on"] and PARAMS["wandb_save_checkpoint"],
                                checkpoint_artifact=CHECKPOINT_ARTIFACT_MIN_LOSS)

            wandb_log_final_result(val_metrics, PARAMS)
            wandb.log_artifact(CHECKPOINT_ARTIFACT_MIN_LOSS)
            wandb.log_artifact(CHECKPOINT_ARTIFACT_MAX_ACC)
            wandb.finish()

            # after everything, so the run does not break
            early_stopper.register_metric(val_loss)
            if early_stopper.should_early_stop():
                early_stopper.reset()
                early_stopper.log_if_stopped()
                break


if __name__ == "__main__":
    measure_execution_time(main)