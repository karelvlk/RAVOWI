from typing import List, Optional, Tuple, Union

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.image_classifiers.datasets.binary_image_dataset_loader import (
    BinaryImageDatasetLoader,
)
from src.image_classifiers.datasets.json_dataset_loader import JsonDatasetLoader

from ..base.base_classification_trainer import BaseClassificationTrainer
from .binary_classifier import BinaryClassifier


class BinaryTrainer(BaseClassificationTrainer):
    def __init__(
        self,
        name: str,
        classes: List[str],
        args: dict,
        data_path: str,
        dataloader_type: str,
        model: Optional[nn.Module] = None,
        train: bool = True,
    ) -> None:
        self.dataloader_type = dataloader_type
        super().__init__(name, classes, args, data_path, model, train)

    def get_model(self) -> BinaryClassifier:
        return BinaryClassifier(self.args)

    def get_loss_fn(self, class_weights: Optional[torch.Tensor] = None) -> nn.BCELoss:
        return nn.BCELoss(weight=class_weights)

    def get_metrics(
        self, pred: torch.Tensor, gold: torch.Tensor, threshold: float = 0.5
    ) -> Tuple[float, float]:
        pred = pred.cpu()
        gold = gold.cpu()
        pred_binary = (pred > threshold).float()
        gold_binary = (gold > threshold).float()

        accu = accuracy_score(gold_binary, pred_binary)
        f1 = f1_score(gold_binary, pred_binary)

        return accu, f1

    def get_dataloader(
        self, _, data_path: str, log_dir: str
    ) -> Union[BinaryImageDatasetLoader, JsonDatasetLoader]:
        if self.dataloader_type == "image":
            return BinaryImageDatasetLoader(1, data_path, log_dir)
        elif self.dataloader_type == "json":
            return JsonDatasetLoader(path=data_path)
        else:
            raise NotImplementedError(
                f"Dataloader for type {self.dataloader_type} is not implemented"
            )
