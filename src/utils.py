import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np
from torch import nn

def save_checkpoint(state, fold=""):
    filename=f"{fold}_checkpoint.pth.tar"
    print("Saving checkpoint...")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer, device):
    # TODO: add and check device
    checkpoint = torch.load(checkpoint_path)

    # load variables
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['step']

    return step


def set_gpu_mode(model):
    # for more than 1 GPU
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!\n")
        model = nn.DataParallel(model)
    else:
        print('Using a single GPU\n')


def initialize_wandb(inputs):
    if inputs.wandb_on:
        wandb.init(name=inputs.name, project=inputs.PROJECT, entity=inputs.ENTITY)
        wandb.config = inputs.wandb


def get_balanced_dataset_sampler(data_loader, train_ids, train_subset):


    binary_labels = [sample[1] for sample in data_loader.dataset.samples]
    class_weights = 1 / np.unique(np.array(binary_labels)[train_ids], return_counts=True)[1] 

    sample_weights = [0] * len(train_ids)


    for idx, (data, label) in enumerate(train_subset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler