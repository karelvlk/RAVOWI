from typing import List, Optional, Union, Tuple
import numpy as np

import torch
from torch import nn

from src.image_classifiers.datasets.json_dataset_loader import JsonDatasetLoader
from src.image_classifiers.datasets.multi_image_dataset_loader import (
    MultiImageDatasetLoader,
)

from ..base.base_classification_trainer import BaseClassificationTrainer
from .singlelabel_classifier import SinglelabelClassifier
from sklearn.metrics import f1_score, accuracy_score
import sklearn

class SinglelabelTrainer(BaseClassificationTrainer):
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

    def get_model(self) -> nn.Module:
        return SinglelabelClassifier(self.args)

    def get_loss_fn(self, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)
    
    def get_metrics(
        self, pred: torch.Tensor, gold: torch.Tensor
    ) -> Tuple[float, float, float]:
        pred = pred.cpu().detach().numpy()  # move to cpu before converting to numpy
        y_true = gold.cpu().detach().numpy()  # move to cpu before converting to numpy
        
        # print('SINGLELABEL PREDS:', pred)
        # print('SINGLELABE GOLD:', gold)

        # Convert softmax outputs to predicted labels
        y_pred = np.argmax(pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        
        f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
        f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average='weighted')

        # Compute Accuracy
        accu = sklearn.metrics.accuracy_score(y_true, y_pred)


        return accu, f1_macro, f1_weighted

    def get_dataloader(
        self, num_classes: int, data_path: str, log_dir: str
    ) -> Union[MultiImageDatasetLoader, JsonDatasetLoader]:
        if self.dataloader_type == "image":
            # Multi is used because single need exact same implementation
            return MultiImageDatasetLoader(num_classes, path=data_path, d=log_dir)
        elif self.dataloader_type == "json":
            return JsonDatasetLoader(path=data_path)
        else:
            raise NotImplementedError(
                f"Dataloader for type {self.dataloader_type} is not implemented"
            )
