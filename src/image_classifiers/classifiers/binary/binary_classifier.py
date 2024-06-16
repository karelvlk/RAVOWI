from typing import Any, Dict

import torch

from ..base.base_classification_model import BaseClassificationModel


class BinaryClassifier(BaseClassificationModel):
    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.base_model(x))
