from typing import Tuple

from torch.utils.data import Dataset
from torchvision import transforms

from src.image_classifiers.datasets.base_dataset_loader import BaseDatasetLoader


class BaseImageDatasetLoader(BaseDatasetLoader):
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
        super().__init__(num_classes, path, d, bs, train_split, val_split, test_split)
        self.train_augmentator = self.get_train_augmentator()
        self.val_augmentator = self.get_val_augmentator()

    def addTransform(
        self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset
    ) -> Tuple[Dataset, Dataset, Dataset]:
        train_dataset.transform = self.train_augmentator
        val_dataset.transform = self.val_augmentator
        test_dataset.transform = self.val_augmentator

        return train_dataset, val_dataset, test_dataset

    def get_train_augmentator(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),  # Flip images horizontally at random
                # Rotate images at random by up to 30 degrees
                transforms.RandomRotation(30),
                transforms.RandomInvert(),
                # Change brightness, contrast, and saturation at random
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomPerspective(),  # Apply random perspective transformation
                # Apply Gaussian blur
                transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
                transforms.ToTensor(),  # Convert images to PyTorch tensors
            ]
        )

    def get_val_augmentator(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
