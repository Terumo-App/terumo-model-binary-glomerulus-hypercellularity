import wandb
import torch
import torch.nn.functional as F
import os
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
import torch
from tqdm import tqdm
import torch.nn.functional as F
import time
from pathlib import Path
from torch import nn
import yaml
from metrics import Metrics

def train_epoch(loader, model, optimizer, loss_fn, scaler, device):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for batch_idx, (data, y) in enumerate(loader):
        data = data.to(device=device)
        y = y.to(device=device)
        targets = F.one_hot(y, num_classes=2)

        # forward
        with torch.cuda.amp.autocast():
            output = model(data)
            # print(targets.float())
            loss = loss_fn(output, targets.float())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * data.size(0)

        scores = torch.sigmoid(output)
        predictions = (scores>0.5).float()
        _, pred = torch.max(predictions, 1)
        
        train_correct += (pred == y ).sum()

    return train_loss, train_correct

def valid_epoch(loader, model, loss_fn=None, device="cuda"):
    num_correct = 0
    valid_loss = 0.0
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
  
            output = model(x)
            scores = torch.sigmoid(output)
            predictions = (scores>0.5).float()
            
            target = F.one_hot(y, num_classes=2)

            if loss_fn:
                loss = loss_fn(output, target.float())
                valid_loss+=loss.item()*x.size(0)

            _, pred = torch.max(predictions, 1)

            num_correct += (pred == y).sum()

            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())


    metrics_result = Metrics()
    metrics_result.compute_metrics(y_true, y_pred)
    
    return valid_loss, num_correct, metrics_result
          


    

def save_checkpoint(model, optimizer, create_timestamp_folder, metric_type, fold=""):
    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    Path(f'./artifacts/{create_timestamp_folder}').mkdir(exist_ok=True)
    filename=f"./artifacts/{create_timestamp_folder}/{fold}_fold_{metric_type}_checkpoint.pth.tar"
    # print("Saving checkpoint...")
    torch.save(state, filename)



def load_checkpoint(path, model, optimizer):
    print("=> Loading checkpoint")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def make_prediction(model, transform, rootdir, device):
    files = os.listdir(rootdir)
    preds = []
    model.eval()

    files = sorted(files, key=lambda x: float(x.split(".")[0]))
    for file in tqdm(files):
        img = Image.open(os.path.join(rootdir, file))
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(img))
            preds.append(pred.item())


    df = pd.DataFrame({'id': np.arange(1, len(preds)+1), 'label': np.array(preds)})
    df.to_csv('submission.csv', index=False)
    model.train()
    print("Done with predictions")


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

def get_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])



def create_timestamp_folder(model_name):
    """
    Create a folder name based on the current timestamp.
    Returns:
        folder_name (str): The name of the folder, in the format 'YYYY-MM-DD-HH-MM-SS'.
    """
    current_time = time.localtime()
    folder_name = time.strftime('%Y-%m-%d-%H-%M-%S', current_time)
    return f'{model_name}_{folder_name}'


def initialize_wandb(inputs, fold, folder_name):
    if inputs['wandb_on']:
        wandb.init(
            name=f'{folder_name}_{fold}', 
            project=inputs['project'],
            config=inputs
            )



def set_gpu_mode(model):
    # for more than 1 GPU
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!\n")
        model = nn.DataParallel(model)
    else:
        print('Using a single GPU\n')



def compute_metrics(targets, predictions):
    """
    Compute various classification metrics including F1 score, precision, and recall.

    Args:
        targets (torch.Tensor): Ground truth labels.
        predictions (torch.Tensor): Predicted labels.

    Returns:
        f1_score (float): F1 score.
        precision (float): Precision.
        recall (float): Recall.
    """
    # Convert tensors to numpy arrays
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()

    # Calculate true positives, false positives, and false negatives
    true_positives = ((predictions == 1) & (targets == 1)).sum()
    false_positives = ((predictions == 1) & (targets == 0)).sum()
    false_negatives = ((predictions == 0) & (targets == 1)).sum()

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    return f1_score, precision, recall


def measure_execution_time(func):
    """
    Measure the execution time of a function.

    Args:
        func (callable): The function to measure execution time for.

    Returns:
        elapsed_time (float): The elapsed time in seconds.
    """
    start_time = time.time()
    func()  # Execute the function
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_min =  elapsed_time / 60

    print(f"Training elapsed Time: {elapsed_time_min:.3f} minutes")
    


def load_training_parameters(filename):
    with open(filename, 'r') as file:
        params = yaml.safe_load(file)
    return params

def wandb_log_final_result(metrics:Metrics, config):

    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=metrics.y_true, preds=metrics.y_pred,
                            class_names=config['classes'])})
    
    # wandb.log({"pr" : wandb.plot.pr_curve(metrics.y_true, metrics.y_pred,
    #             labels=None, classes_to_plot=None)})

    # wandb.log({"ROC" : wandb.plot.roc_curve(metrics.y_true, metrics.y_pred,
    #                         labels=config['classes'])})

    wandb.summary['test_accuracy'] = metrics.accuracy
    wandb.summary['test_precision'] = metrics.precision
    wandb.summary['test_recall'] = metrics.recall
    wandb.summary['test_fscore'] = metrics.fscore
    wandb.summary['test_kappa'] = metrics.kappa
    # wandb.log({
    #     'final_accuracy': metrics.accuracy,
    #     'final_precision': metrics.precision,
    #     'final_recall': metrics.recall,
    #     'final_fscore': metrics.fscore,
    #     'final_kappa': metrics.kappa           
    #     })    
