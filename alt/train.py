import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Net
from utils import check_accuracy, load_checkpoint, save_checkpoint, make_prediction, get_balanced_dataset_sampler, get_train_transform, get_test_transform
import config
from dataset import ImageDataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    for batch_idx, (data, targets) in enumerate(tqdm(loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        # targets = targets.to(device=device)
        targets = F.one_hot(targets.to(device=device), num_classes=2)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            # print(targets.float())
            loss = loss_fn(scores, targets.float())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def main(train_loader, val_loader):

    print(f'Device is {config.DEVICE}')
    loss_fn = nn.CrossEntropyLoss()
    model = Net(net_version="b0", num_classes=2).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    # make_prediction(model, config.val_transforms, './data/results', config.DEVICE)
    check_accuracy(val_loader, model, config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        print("#"*60)
        print(f"Starting epoch: {epoch+1}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)
        check_accuracy(val_loader, model, config.DEVICE)
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
        print("\n\n")

if __name__ == "__main__":

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
        test_loader = DataLoader(test_subset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True )


        print(f'FOLD {fold +1}')
        print('--------------------------------')
        main(train_loader, test_loader)