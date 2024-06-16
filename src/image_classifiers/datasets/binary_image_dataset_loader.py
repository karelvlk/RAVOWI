import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.image_classifiers.datasets.base_image_dataset_loader import (
    BaseImageDatasetLoader,
)


class BinaryClassificationDataset(Dataset):
    def __init__(
        self,
        num_classes: int,
        data: List[Dict[str, Any]],
        d: str,
        ls: float = 0.3,
        transform=None,
    ):
        self.data = data
        self.dir = d
        self.transform = transform
        self.shown = 0
        self.num_classes = num_classes
        self.ls = ls

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]

        image = cv2.imread(item["path"])

        if image is None:
            logging.warning(f"Failed to load image at path: {item['path']}")
            item["annotation"] = self.ls
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        annotation = torch.tensor(
            1 - self.ls if item["annotation"] == 1 else self.ls, dtype=torch.float32
        )
        if self.transform:
            image = self.transform(image)

        return image, annotation


class BinaryImageDatasetLoader(BaseImageDatasetLoader):
    def __init__(
        self,
        num_classes: int,
        path: str,
        d: str,
        bs: int = 64,
        train_split: float = 0.85,
        val_split: float = 0.15,
        test_split: float = 0.0,
    ) -> None:
        super().__init__(num_classes, path, d, bs, train_split, val_split, test_split)

    def getFullDataset(
        self,
        nc: int,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        directory: str,
    ) -> Tuple[
        BinaryClassificationDataset,
        BinaryClassificationDataset,
        BinaryClassificationDataset,
    ]:
        return (
            BinaryClassificationDataset(nc, train_data, directory),
            BinaryClassificationDataset(nc, val_data, directory),
            BinaryClassificationDataset(nc, test_data, directory),
        )
