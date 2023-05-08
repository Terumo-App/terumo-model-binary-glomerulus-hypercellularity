import torch


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
