from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from src.image_classifiers.datasets.base_dataset_loader import BaseDatasetLoader


class JSON_Dataset(Dataset):
    def __init__(self, json_data: List[Dict[str, float]]) -> None:
        self.data = json_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        features = torch.tensor(item["features"], dtype=torch.float32)
        target = torch.tensor(item["target"], dtype=torch.float32)

        return features, target


class JsonDatasetLoader(BaseDatasetLoader):
    def __init__(
        self,
        path: str,
        bs: int = 16,
        train_split: float = 0.85,
        val_split: float = 0.15,
        test_split: float = 0.0,
    ) -> None:
        super().__init__(1, path, "no_dir", bs, train_split, val_split, test_split)

    def getFullDataset(
        self, nc: int, train_data: List[Dict[str, float]], val_data: List[Dict[str, float]], test_data: List[Dict[str, float]], dir: str
    ) -> Tuple[JSON_Dataset, JSON_Dataset, JSON_Dataset]:
        return (
            JSON_Dataset(train_data),
            JSON_Dataset(val_data),
            JSON_Dataset(test_data)
        )
