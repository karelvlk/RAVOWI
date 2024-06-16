import argparse
import logging
import os
import shutil
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class YOLODataPreprocessor:
    def __init__(
        self,
        base_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "images")
        self.labels_dir = os.path.join(base_dir, "labels")
        self.temp_dir = os.path.join(base_dir, "temp")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def preprocess(self) -> None:
        train_count, valid_count, test_count = self.split_data()
        total_count = train_count + valid_count + test_count
        logging.debug(
            f"Preprocessed splitted dataset train_count: {train_count} ({round((train_count / total_count) * 100, 2)}%)"
        )
        logging.debug(
            f"Preprocessed splitted dataset valid_count: {valid_count} ({round((valid_count / total_count) * 100, 2)}%)"
        )
        logging.debug(
            f"Preprocessed splitted dataset test_count: {test_count} ({round((test_count / total_count) * 100, 2)}%)"
        )

    def visualize_label_distribution(
        self,
        df: pd.DataFrame,
        train_idx: pd.Index,
        val_idx: pd.Index,
        test_idx: pd.Index,
    ) -> None:
        _, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        df.loc[train_idx, "class_id"].value_counts().plot(
            kind="bar", ax=axes[0], title="Train Set"
        )
        df.loc[val_idx, "class_id"].value_counts().plot(
            kind="bar", ax=axes[1], title="Validation Set"
        )
        df.loc[test_idx, "class_id"].value_counts().plot(
            kind="bar", ax=axes[2], title="Test Set"
        )
        plots_dir = os.path.join(self.base_dir, "plots")
        logging.debug(f"Saving plots to {plots_dir}")
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "label_distribution.png"))
        plt.show()

    def get_label_data(self) -> pd.DataFrame:
        labels = [
            os.path.join(self.labels_dir, label)
            for label in os.listdir(self.labels_dir)
            if label.endswith(".txt")
        ]
        data = []
        for label_file in labels:
            with open(label_file, "r") as file:
                lines = file.readlines()
                for line in lines:
                    class_id = line.split()[0]
                    if class_id != "0":
                        logging.debug(label_file)

                    data.append(
                        {
                            "image": os.path.basename(label_file).replace(".txt", ""),
                            "class_id": class_id,
                        }
                    )
        return pd.DataFrame(data)

    def is_already_split(self) -> Tuple[bool, int, int, int]:
        train_count = val_count = test_count = 0
        is_split = True

        for split in ["train", "val", "test"]:
            split_img_dir = os.path.join(self.base_dir, split, "images")
            if not os.path.exists(split_img_dir) or not os.listdir(split_img_dir):
                is_split = False
                break
            else:
                count = len(
                    [
                        file
                        for file in os.listdir(split_img_dir)
                        if file.endswith(".jpg")
                    ]
                )
                if split == "train":
                    train_count = count
                elif split == "val":
                    val_count = count
                elif split == "test":
                    test_count = count

        return is_split, train_count, val_count, test_count

    def split_data(self) -> Tuple[int, int, int]:
        is_split, train_count, val_count, test_count = self.is_already_split()
        if is_split:
            logging.debug("Data is already split. Skipping splitting.")
            return train_count, val_count, test_count

        df = self.get_label_data()

        # Splitting data
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=1 - self.train_ratio,
            stratify=df["class_id"],
            random_state=42,
        )

        val_idx, test_idx = train_test_split(
            test_idx,
            test_size=self.test_ratio / (self.test_ratio + self.val_ratio),
            stratify=df.loc[test_idx, "class_id"],
            random_state=42,
        )

        def copy_files(idx: pd.Index, split: str) -> None:
            os.makedirs(os.path.join(self.base_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.base_dir, split, "labels"), exist_ok=True)
            with open(os.path.join(self.temp_dir, f"{split}_data.txt"), "w") as f:
                for i in idx:
                    img_name = df.loc[i, "image"]
                    shutil.copy(
                        os.path.join(self.images_dir, img_name + ".jpg"),
                        os.path.join(self.base_dir, split, "images", img_name + ".jpg"),
                    )
                    shutil.copy(
                        os.path.join(self.labels_dir, img_name + ".txt"),
                        os.path.join(self.base_dir, split, "labels", img_name + ".txt"),
                    )
                    f.write(f"{img_name}\n")

        os.makedirs(self.temp_dir, exist_ok=True)

        copy_files(train_idx, "train")
        copy_files(val_idx, "val")
        copy_files(test_idx, "test")

        # Visualization
        self.visualize_label_distribution(df, train_idx, val_idx, test_idx)

        return len(train_idx), len(val_idx), len(test_idx)


# parser = argparse.ArgumentParser(description="YOLO Data Preprocessor")
# parser.add_argument("--dataset_path", type=str, help="Path to the dataset")

# args = parser.parse_args()

# preprocessor = YOLODataPreprocessor(args.dataset_path)
# train_count, val_count, test_count = preprocessor.split_data()
# logging.info(
#     f"Dataset on path: {args.dataset_path} preprocessed\nTraining samples: {train_count}, Validation samples: {val_count}, Testing samples: {test_count}"
# )
