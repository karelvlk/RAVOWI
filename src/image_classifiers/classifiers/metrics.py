import torch
from torch import Tensor


def accuracy_multilabel(
    outputs: Tensor, target: Tensor, threshold: float = 0.5
) -> float:
    with torch.no_grad():
        pred = (outputs > threshold).float()
        correct = (pred == target).float().sum()
        acc = correct / (target.size(0) * target.size(1))
    return acc.item()


def onehot_to_indices(onehot_tensor: Tensor) -> Tensor:
    return torch.argmax(onehot_tensor, dim=1)


def categorical_accuracy(predictions: Tensor, targets: Tensor) -> float:
    target_indices = onehot_to_indices(targets)
    predicted_indices = torch.argmax(predictions, dim=1)
    correct = (predicted_indices == target_indices).sum().item()
    accuracy = correct / target_indices.size(0)
    return accuracy


def categorical_precision(
    predictions: Tensor, targets: Tensor, class_index: int
) -> float:
    target_indices = onehot_to_indices(targets)
    predicted_indices = torch.argmax(predictions, dim=1)

    true_positives = (
        ((predicted_indices == class_index) & (target_indices == class_index))
        .sum()
        .item()
    )
    predicted_positives = (predicted_indices == class_index).sum().item()

    # Adding epsilon to avoid division by zero
    precision = true_positives / (predicted_positives + 1e-8)
    return precision


def categorical_recall(predictions: Tensor, targets: Tensor, class_index: int) -> float:
    target_indices = onehot_to_indices(targets)
    predicted_indices = torch.argmax(predictions, dim=1)

    true_positives = (
        ((predicted_indices == class_index) & (target_indices == class_index))
        .sum()
        .item()
    )
    actual_positives = (target_indices == class_index).sum().item()

    # Adding epsilon to avoid division by zero
    recall = true_positives / (actual_positives + 1e-8)
    return recall


def categorical_f1(predictions: Tensor, targets: Tensor, class_index: int) -> float:
    precision = categorical_precision(predictions, targets, class_index)
    recall = categorical_recall(predictions, targets, class_index)

    # Adding epsilon to avoid division by zero
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


def macro_f1(predictions: Tensor, targets: Tensor) -> float:
    num_classes = targets.size(1)
    f1_scores = []

    for class_index in range(num_classes):
        f1_scores.append(categorical_f1(predictions, targets, class_index))

    return sum(f1_scores) / num_classes
