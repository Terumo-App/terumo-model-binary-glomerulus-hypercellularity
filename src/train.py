import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Net
from utils import valid_epoch, load_checkpoint, save_checkpoint, make_prediction, get_balanced_dataset_sampler, get_train_transform, get_test_transform, train_epoch, create_timestamp_folder, initialize_wandb, set_gpu_mode, measure_execution_time

import sys
import config
from dataset import ImageDataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import wandb

# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
# But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.
torch.backends.cudnn.benchmark = True


def main():

    print(f'Device is {config.DEVICE}')

    initialize_wandb(config)

    artifact_folder = create_timestamp_folder(config.MODEL_NAME)
    skfold = StratifiedKFold(n_splits=5, shuffle=True)
    data_loader = ImageDataLoader(config.DATA_DIR)
    binary_labels = [sample[1] for sample in data_loader.dataset.samples]
    for fold, (train_ids, test_ids) in enumerate(skfold.split(data_loader.dataset, binary_labels)):

        train_subset = Subset(data_loader.dataset, train_ids)
        train_subset.transform = get_train_transform()
        sampler = get_balanced_dataset_sampler(data_loader, train_ids, train_subset)
        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=config.NUM_WORKERS)

        test_subset = Subset(data_loader.dataset, test_ids) 
        test_subset.transform = get_test_transform()
        val_loader = DataLoader(test_subset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True )


        print(f'Fold  {fold +1}')

        loss_fn = nn.CrossEntropyLoss()
        model = Net(net_version="b0", num_classes=2).to(config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scaler = torch.cuda.amp.GradScaler()
        set_gpu_mode(model)

        if config.LOAD_MODEL:
            load_checkpoint(torch.load(config.CHECKPOINT_TO_BE_LOADED), model, optimizer)

        valid_epoch(val_loader, model, loss_fn, config.DEVICE)
        
        max_val_accuracy, min_val_loss = 0, sys.maxsize
        for epoch in range(config.NUM_EPOCHS):
                   


            train_metrics = train_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)
            test_metrics = valid_epoch(val_loader, model, loss_fn, config.DEVICE)

            train_loss, train_correct = train_metrics
            test_loss, test_correct, f1_score = test_metrics

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(val_loader.sampler)
            test_acc = test_correct / len(val_loader.sampler) * 100
            test_f1_score = f1_score / len(val_loader.sampler) * 100

            print(f"Epoch:{epoch + 1}/{config.NUM_EPOCHS} AVG Training Loss:{train_loss:.3f}, AVG Test Loss:{test_loss:.3f}, AVG Training Acc {train_acc:.2f}%, AVG Test Acc {test_acc:.2f}%, F1-Score {test_f1_score:.2f}")
            
            if config.WANDB_ON:
                wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'f1_score': test_f1_score
                })

            if max_val_accuracy < test_acc:
                max_val_accuracy = test_acc
                save_checkpoint(model, optimizer, artifact_folder, 'max_acc', fold)

            if min_val_loss > test_loss:
                min_val_loss = test_loss
                save_checkpoint(model, optimizer, artifact_folder, 'min_loss', fold)
            


if __name__ == "__main__":
        measure_execution_time(main())
        