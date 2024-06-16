from typing import Any, Dict

import torch
import torch.nn.functional as F

from ..base.base_classification_model import BaseClassificationModel


class SinglelabelClassifier(BaseClassificationModel):
    def __init__(self, args: Dict[str, Any]):
        super().__init__(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.base_model(x), dim=1)
