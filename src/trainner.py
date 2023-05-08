from metrics import compute_metrics

import torch

def train(model, train_loader, optimizer, scheduler, criterion, scaler, device):
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
    # wandb.log({
    #     'train_loss': running_loss / len(train_loader),
    #     'train_accuracy': accuracy
    # })