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
from config import settings

# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
# But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.
torch.backends.cudnn.benchmark = True
opt = BaseOptions().parse()
PARAMS = load_training_parameters(opt.config_file)
wandb.login(key=PARAMS['wandb_key'])

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
    for fold, (train_ids, val_ids) in enumerate(skfold.split(full_dataset_train_mode, binary_labels)):

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

        # valid_epoch(val_loader, model, loss_fn, settings.config.DEVICE)
        
        max_val_accuracy, min_val_loss = 0, sys.maxsize
        for epoch in range(PARAMS['num_epochs']):
            train_metrics = train_epoch(train_loader, model, optimizer, loss_fn, scaler, settings.config.DEVICE)
            test_metrics = valid_epoch(val_loader, model, loss_fn, settings.config.DEVICE)

            train_loss, train_correct = train_metrics
            test_loss, test_correct, metrics = test_metrics

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(val_loader.sampler)
            test_acc = test_correct / len(val_loader.sampler) * 100
  

            print(f"Epoch:{epoch + 1}/{PARAMS['num_epochs']} AVG Training Loss:{train_loss:.3f}, AVG Test Loss:{test_loss:.3f}, AVG Training Acc {train_acc:.2f}%, AVG Test Acc {test_acc:.2f}% , Test F1 Score {metrics.fscore*100:.2f}%")
            
            if PARAMS['wandb_on']:
                wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': test_loss,
                'val_acc': test_acc,
                'val_fscore': metrics.fscore*100,
                })

            if max_val_accuracy < test_acc:
                max_val_accuracy = test_acc
                save_checkpoint(model, optimizer, artifact_folder, 'max_acc', fold)

            if min_val_loss > test_loss:
                min_val_loss = test_loss
                save_checkpoint(model, optimizer, artifact_folder, 'min_loss', fold)

        if PARAMS['wandb_on']:
            _, _, metrics = valid_epoch(val_loader, model, loss_fn, settings.config.DEVICE)
            wandb_log_final_result(metrics, PARAMS)
            wandb.finish()



if __name__ == "__main__":
        measure_execution_time(main)
        