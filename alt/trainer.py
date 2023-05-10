import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Net
from utils import valid_epoch, load_checkpoint, save_checkpoint, make_prediction, get_balanced_dataset_sampler, get_train_transform, get_test_transform, train_epoch
import config
from dataset import ImageDataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from train import train_fn



def main():

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


        print(f'FOLD {fold +1}')
        print('--------------------------------')
        print(f'Device is {config.DEVICE}')

        loss_fn = nn.CrossEntropyLoss()
        model = Net(net_version="b0", num_classes=2).to(config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scaler = torch.cuda.amp.GradScaler()

        if config.LOAD_MODEL:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

        # make_prediction(model, config.val_transforms, './data/results', config.DEVICE)
        valid_epoch(val_loader, model, config.DEVICE)

        for epoch in range(config.NUM_EPOCHS):
            print("#"*60)
            print(f"Starting epoch: {epoch+1}")
            train_fn(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)
            valid_epoch(val_loader, model, config.DEVICE)
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)


if __name__ == "__main__":


        main(train_loader, test_loader)