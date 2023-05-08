"""
Main module for training the networks for gender estimation.
Author: Bernardo Silva (https://github.com/bpmsilva)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision

import wandb

from configs import Inputs
from utils.data import RadiographSexDataset
from utils.augmentations import get_transforms
from utils.lambda_schedulers import linear_warmup



def initialize_wandb(inputs):
    wandb.init(name=inputs.name, project=inputs.PROJECT, entity=inputs.ENTITY)
    wandb.config = inputs.wandb

def get_classification_model(model_name, num_classes):
    if model_name == 'resnet-50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet-b0':
        model = torchvision.models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet-b4':
        model = torchvision.models.efficientnet_b4(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet-b7':
        model = torchvision.models.efficientnet_b7(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise Exception(f'Model {model_name} is not supported')
    return model

def load_checkpoint(checkpoint_path, model, optimizer, device):
    # TODO: add and check device
    checkpoint = torch.load(checkpoint_path)

    # load variables
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['step']

    return step

def compute_metrics(outputs, labels):
    # convert outputs to the predicted classes
    _, pred = torch.max(outputs, 1)

    # compare predictions to true label
    total = len(labels)
    true_postives = pred.eq(labels.data.view_as(pred)).sum().item()
    accuracy = true_postives / len(labels)

    return {
        'tp': true_postives,
        'accuracy': accuracy,
        'total': total
    }

def train_step(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    scaler,
    device
):
    running_loss, tp, total = 0, 0, 0
    for imgs, labels in train_loader:
        # put model in training mode
        model.train()
        # send images and labels to device
        imgs, labels = imgs.to(device), labels.to(device)

        # feedforward and loss with mixed-precision
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # TODO: check if this output is logits, probabilities or log of probabilities
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        # sum up the loss
        running_loss += loss.item() * len(imgs)

        # backpropagation with mixed precision training
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        scheduler.step()

        metrics = compute_metrics(outputs, labels)
        tp += metrics['tp']
        total += metrics['total']

    accuracy = tp / total
    print(f'Training loss: {running_loss / len(train_loader):.5f}')
    print(f'Training accuracy: {100*accuracy:.2f} (%)')

    # wandb log
    wandb.log({
        'train_loss': running_loss / len(train_loader),
        'train_accuracy': accuracy
    })

def val_step(
    model,
    val_loader,
    criterion,
    device
):
    with torch.no_grad():
        running_loss, tp, total = 0, 0, 0
        for imgs, labels in val_loader:
            # put model in evaluation mode
            model.eval()
            # send images and labels to device
            imgs, labels = imgs.to(device), labels.to(device)

            # feedforward
            # TODO: check if I should add mixed-precision here
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            # sum up the loss
            running_loss += loss.item() * len(imgs)

            metrics = compute_metrics(outputs, labels)
            tp += metrics['tp']
            total += metrics['total']

        accuracy = tp / total
        print(f'Validation loss: {running_loss / len(val_loader):.5f}')
        print(f'Validation accuracy: {100*accuracy:.2f} (%)')

        # wandb log
        val_loss = running_loss / len(val_loader)
        wandb.log({
            'val_loss': val_loss,
            'val_accuracy': accuracy
        })

        return accuracy, val_loss

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint...")
    torch.save(state, filename)

def train(
    name,
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    epochs,
    save_model=True,
    load_model=None
):
    # not that important, but apparently gives performance boost
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device is {device}')

    if load_model:
        step = load_checkpoint(load_model, model, optimizer, device)
    else:
        step = 0

    # for more than 1 GPU
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!\n")
        model = nn.DataParallel(model)
    else:
        print('Using a single GPU\n')
    model.to(device)

    # mixed-precision training
    scaler = torch.cuda.amp.GradScaler()

    # training loop
    max_val_accuracy, min_val_loss = 0, 10000000000
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')

        train_step(model, train_loader, optimizer, scheduler, criterion, scaler, device)
        print()
        val_accuracy, val_loss = val_step(model, valid_loader, criterion, device)
        print()

        if max_val_accuracy < val_accuracy:
            print(f'Accuracy increased from {max_val_accuracy:.4f}' + \
                  f' to {val_accuracy:.4f} ({epoch}/{epochs})')

            max_val_accuracy = val_accuracy
            if save_model:
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step
                }
                save_checkpoint(checkpoint, filename=f'checkpoint-{name}-max-acc.pth.tar')

        if min_val_loss > val_loss:
            print(f'Validation loss decreased from {min_val_loss:.2f}' + \
                  f' to {val_loss:.2f} ({epoch}/{epochs})')

            min_val_loss = val_loss
            if save_model:
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step
                }
                save_checkpoint(checkpoint, filename=f'checkpoint-{name}-min-loss.pth.tar')

def main():
    inputs = Inputs()

    initialize_wandb(inputs)

    folds = {
        'train': [fold for fold in range(1, inputs.num_folds + 1) if fold != inputs.val_fold],
        'val':   [inputs.val_fold]
    }

    # get the datasets
    datasets, dataloaders = dict(), dict()
    for subset in ['train', 'val']:
        datasets[subset] = RadiographSexDataset(
            root_dir=inputs.DATASET_DIR,
            fold_nums=folds[subset],
            transforms=get_transforms(inputs, subset=subset),
            crop_side=inputs.crop_side,
            border=inputs.remove_border
        )

        dataloaders[subset] = DataLoader(
            datasets[subset],
            batch_size=inputs.batches_and_lr[f'{subset}_batch_size'],
            shuffle=subset=='train',
            num_workers=inputs.NUM_WORKERS
        )

    print(inputs)
    print(f'\nModel is {inputs.model_name}')
    model = get_classification_model(inputs.model_name, 2)

    # optimizer and scheduler
    optimizer = inputs.OPTIMIZER(model.parameters(), lr=inputs.lr)
    warmup_steps = len(dataloaders['train']) * inputs.WARMUP_EPOCHS
    scheduler = linear_warmup(optimizer, warmup_steps)

    train(
        name=inputs.name,
        model=model,
        criterion=inputs.CRITERION,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=dataloaders['train'],
        valid_loader=dataloaders['val'],
        epochs=inputs.EPOCHS,
    )

if __name__ == '__main__':
    main()
