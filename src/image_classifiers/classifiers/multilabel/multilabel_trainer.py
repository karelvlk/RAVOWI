from typing import List, Optional, Union, Tuple
import numpy as np
import torch
from torch import nn

from src.image_classifiers.datasets.json_dataset_loader import JsonDatasetLoader
from src.image_classifiers.datasets.multi_image_dataset_loader import (
    MultiImageDatasetLoader,
)

from ..base.base_classification_trainer import BaseClassificationTrainer
from .multilabel_classifier import MultilabelClassifier
from sklearn.metrics import f1_score, accuracy_score
import sklearn

class MultilabelTrainer(BaseClassificationTrainer):
    def __init__(
        self,
        name: str,
        classes: List[str],
        args: dict,
        data_path: str,
        dataloader_type: str,
        model: Optional[nn.Module] = None,
        train: bool = True,
    ):
        self.reduction = args["reduction"] if "reduction" in args else "mean"
        self.dataloader_type = dataloader_type
        super().__init__(name, classes, args, data_path, model, train)

    def get_model(self) -> nn.Module:
        return MultilabelClassifier(self.args)
    
    def get_metrics(
        self, pred: torch.Tensor, gold: torch.Tensor
    ) -> Tuple[float, float, float]:
        pred = pred.cpu().detach().numpy()  # move to cpu before converting to numpy
        y_true = gold.cpu().detach().numpy()  # move to cpu before converting to numpy

        threshold = 0.5
        y_pred = (pred > threshold).astype(int)
        
        accu = sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
        f1_macro = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=1)
        f1_samples = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=1)
        
        # print('+---------------------')
        # print('| My Accuracy: {0}'.format(accu))
        # print('| Exact Match Ratio: {0}'.format(sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))
        # print('| Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_true, y_pred))) 
        # print('| Recall: {0}'.format(sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=1))) 
        # print('| Precision: {0}'.format(sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=1)))
        # print('| F1 Measure Macro: {0}'.format(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=1))) 
        # print('| F1 Measure Weighted: {0}'.format(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=1))) 
        # print('| F1 Measure Micro: {0}'.format(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=1))) 
        # print('| F1 Measure Samples: {0}'.format(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=1))) 
        # print('+---------------------')
        
        return accu, f1_macro, f1_samples


    def get_loss_fn(
        self, class_weights: Optional[torch.Tensor] = None
    ) -> torch.nn.Module:
        return nn.BCELoss(weight=class_weights, reduction=self.reduction)

    def get_dataloader(
        self, num_classes: int, data_path: str, log_dir: str
    ) -> Union[MultiImageDatasetLoader, JsonDatasetLoader]:
        if self.dataloader_type == "image":
            return MultiImageDatasetLoader(num_classes, path=data_path, d=log_dir)
        elif self.dataloader_type == "json":
            return JsonDatasetLoader(path=data_path)
        else:
            raise NotImplementedError(
                f"Dataloader for type {self.dataloader_type} is not implemented"
            )
