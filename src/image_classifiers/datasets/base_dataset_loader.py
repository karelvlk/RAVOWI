from typing import Any, Dict, List, Tuple

import orjson
from torch.utils.data import DataLoader, Dataset, random_split


class BaseDatasetLoader(Dataset):
    def __init__(
        self,
        num_classes: int,
        path: str,
        d: str,
        bs: int = 16,
        train_split: float = 0.85,
        val_split: float = 0.15,
        test_split: float = 0.0,
    ) -> None:
        self.num_classes = num_classes
        self.path = path
        self.bs = 16
        self.dir = d
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def getFullDataset(
        self, nc: int, data: List[Dict[str, Any]], directory: str
    ) -> Dataset:
        # return BinaryClassificationDataset(nc, data, dir)
        raise NotImplementedError()

    def addTransform(
        self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset
    ) -> Tuple[Dataset, Dataset, Dataset]:
        return train_dataset, val_dataset, test_dataset

    def __call__(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # Load the data
        with open(self.path, "r") as train_f, open(
            self.path.replace("train", "val"), "r"
        ) as val_f, open(self.path.replace("train", "test"), "r") as test_f:
            train_data = orjson.loads(train_f.read())
            val_data = orjson.loads(val_f.read())
            test_data = orjson.loads(test_f.read())

        # Calculate the number of samples in each set
        num_samples = len(train_data) + len(val_data) + len(test_data)
        train_size = int(self.train_split * num_samples)
        val_size = int(self.val_split * num_samples)
        test_size = num_samples - train_size - val_size

        # Create the Datasets
        # full_dataset = self.getFullDataset(self.num_classes, data, self.dir)
        # train_dataset, val_dataset, test_dataset = random_split(
        #     full_dataset, [train_size, val_size, test_size]
        # )
        train_dataset, val_dataset, test_dataset = self.getFullDataset(
            self.num_classes, train_data, val_data, test_data, self.dir
        )

        train_dataset, val_dataset, test_dataset = self.addTransform(
            train_dataset, val_dataset, test_dataset
        )

        # Create the DataLoaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.bs, shuffle=True, num_workers=0
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.bs, shuffle=True, num_workers=0
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.bs, shuffle=True, num_workers=0
        )

        return train_dataloader, val_dataloader, test_dataloader
