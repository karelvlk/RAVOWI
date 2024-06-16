import os
import logging
from typing import List, Tuple, Union

import numpy as np
import torch

from src.image_classifiers.categorizer.image_categorization_postrocessor import (
    ImageCategorizationPostrocessor,
)
from src.image_classifiers.base_image_classifier import BaseImageClassifier
from src.settings import cfg

UNIQUE_CLASSES: List[str] = [
    "potted plant",
    "flowers",
    "railing",
    "skyscraper",
    "bicycle",
    "sign",
    "bench",
    "sky",
    "river water",
    "car",
    "boat",
    "road",
    "train",
    "building",
    "grass",
    "airplane",
    "fence",
    "skateboard",
    "sidewalk",
    "snow",
    "house",
    "person",
    "plant",
    "snowy mountain",
    "floor",
    "gate",
    "stones",
    "bridge",
    "trees",
    "fire hydrant",
    "field",
    "plants",
    "snowy ground",
    "path",
    "bushes",
    "sand beach",
    "sand",
    "waterfall",
    "crosswalk",
    "snowboard",
    "hedge",
    "sea water",
    "horse",
    "sheep",
    "rock",
    "buildings",
    "palm tree",
    "motorcycle",
    "stop sign",
    "mountain",
    "traffic light",
    "water fountain",
    "truck",
    "rocky mountain",
    "streetlight",
    "rocks",
    "kite",
    "water",
    "tree",
    "bus",
    "wall",
    "tower",
    "stone",
    "swimming pool",
    "ground",
    "umbrella",
    "balcony",
    "land",
]

UNIQUE_CLASSES2: List[str] = [
    "tree",
    "sea water",
    "skateboard",
    "land",
    "water",
    "plants",
    "stop sign",
    "horse",
    "airplane",
    "motorcycle",
    "ground",
    "sheep",
    "boat",
    "truck",
    "path",
    "bicycle",
    "traffic light",
    "field",
    "plant",
    "stones",
    "building",
    "stone",
    "floor",
    "train",
    "kite",
    "bench",
    "mountain",
    "snowboard",
    "hedge",
    "palm tree",
    "streetlight",
    "snowy ground",
    "water fountain",
    "person",
    "sand beach",
    "wall",
    "fire hydrant",
    "balcony",
    "flowers",
    "umbrella",
    "sky",
    "rocky mountain",
    "fence",
    "gate",
    "fireplace",
    "bushes",
    "sign",
    "crosswalk",
    "rock",
    "buildings",
    "sand",
    "snow",
    "swimming pool",
    "road",
    "potted plant",
    "bus",
    "rocks",
    "snowy mountain",
    "river water",
    "tower",
    "plate",
    "bridge",
    "skyscraper",
    "house",
    "car",
    "grass",
    "trees",
    "railing",
    "sidewalk",
    "waterfall",
]
UNIQUE_CLASSES_MLP_NEW: List[str] = [
    "fence",
    "water",
    "field",
    "snow",
    "rock",
    "fire hydrant",
    "waterfall",
    "airplane",
    "grass",
    "sheep",
    "crosswalk",
    "stop sign",
    "train",
    "rocky mountain",
    "land",
    "snowy ground",
    "sand beach",
    "motorcycle",
    "tree",
    "plants",
    "bus",
    "snowboard",
    "house",
    "swimming pool",
    "building",
    "road",
    "kite",
    "floor",
    "balcony",
    "sand",
    "gate",
    "bridge",
    "rocks",
    "streetlight",
    "traffic light",
    "river water",
    "mountain",
    "umbrella",
    "sidewalk",
    "skateboard",
    "sign",
    "buildings",
    "hedge",
    "skyscraper",
    "bench",
    "trees",
    "bicycle",
    "path",
    "railing",
    "tower",
    "potted plant",
    "car",
    "wall",
    "person",
    "boat",
    "truck",
    "flowers",
    "water fountain",
    "stone",
    "bushes",
    "plant",
    "sky",
    "palm tree",
    "snowy mountain",
    "horse",
    "sea water",
    "ground",
    "stones",
]

UNIQUE_CLASSES_ONEHOT: List[str] = [
    "stop sign",
    "trees",
    "palm tree",
    "swimming pool",
    "bench",
    "snow",
    "road",
    "snowy ground",
    "ground",
    "motorcycle",
    "field",
    "balcony",
    "flowers",
    "rocks",
    "traffic light",
    "crosswalk",
    "wall",
    "train",
    "path",
    "buildings",
    "hedge",
    "building",
    "stones",
    "boat",
    "bus",
    "waterfall",
    "gate",
    "sea water",
    "floor",
    "fire hydrant",
    "bicycle",
    "airplane",
    "horse",
    "streetlight",
    "kite",
    "umbrella",
    "snowy mountain",
    "truck",
    "snowboard",
    "rocky mountain",
    "plants",
    "sky",
    "sand beach",
    "tree",
    "mountain",
    "sign",
    "tower",
    "grass",
    "bushes",
    "river water",
    "sidewalk",
    "stone",
    "water fountain",
    "person",
    "railing",
    "bridge",
    "sand",
    "rock",
    "skyscraper",
    "sheep",
    "potted plant",
    "fence",
    "house",
    "water",
    "plant",
    "car",
    "skateboard",
    "land",
]


class ImageCategorizer(BaseImageClassifier):
    def __init__(self) -> None:
        super().__init__()

        self.image_categorizer, self.classes, _ = self.setup_classifier(
            os.path.join(cfg["CATEGORY_BOX_MODEL"], "best.pt"),
            os.path.join(cfg["CATEGORY_BOX_MODEL"], "cfg.yaml"),
            "multi",
        )
        self.image_onehot_categorizer, self.classes_onehot, _ = self.setup_classifier(
            os.path.join(cfg["CATEGORY_CLS_MODEL"], "best.pt"),
            os.path.join(cfg["CATEGORY_CLS_MODEL"], "cfg.yaml"),
            "multi",
        )

        self.postprocessor = ImageCategorizationPostrocessor()

    def __call__(
        self, object_detection_response: dict, img_size: Tuple[int, int]
    ) -> List[str]:
        area = img_size[0] * img_size[1]
        category, category_onehot = self.make_prediction(
            object_detection_response["classes"],
            object_detection_response["boxes"],
            img_size,
        )

        logging.debug(
            f"raw category prediction: {category}; raw category_onehot prediction: {category_onehot}"
        )

        combined_cats = category  # + [x for x in category_onehot if x not in category]
        # if len(combined_cats) == 0:
        #     combined_cats = ["unknown"]

        return self.postprocessor.category_postprocessing(
            combined_cats, object_detection_response, area
        )

    def make_prediction(
        self, classes: List[str], boxes: List[List[float]], img_size: Tuple[int, int]
    ) -> Tuple[List[str], List[str]]:
        features = self.preprocess_inference_data(
            classes, boxes, UNIQUE_CLASSES, img_size
        )
        features_onehot = self.preprocess_inference_data_onehot(
            classes, UNIQUE_CLASSES_ONEHOT
        )

        predictions = np.array(self._predict(self.image_categorizer, features))[0]
        predictions_onehot = np.array(
            self._predict(self.image_onehot_categorizer, features_onehot)
        )[0]

        filtered_classes = self.classes[predictions.astype(bool)]
        filtered_classes_onehot = self.classes_onehot[predictions_onehot.astype(bool)]

        return list(filtered_classes), list(filtered_classes_onehot)

    def _predict(self, model: torch.nn.Module, features: np.ndarray) -> np.ndarray:
        device = "cpu"
        features_tensor = (
            torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            outputs = model(features_tensor)

        outputs = outputs.cpu()
        threshold = 0.5
        predictions = (outputs > threshold).float().numpy()
        return predictions

    @staticmethod
    def preprocess_inference_data(
        classes: List[str],
        boxes: List[List[float]],
        unique_classes: List[str],
        img_size: Tuple[int, int],
    ) -> List[float]:
        class_features = [int(classes.count(cls) > 0) for cls in unique_classes]

        bbox_features = []
        for cls in unique_classes:
            cls_indices = [i for i, x in enumerate(classes) if x == cls]
            cls_bboxes = [boxes[i][:5] for i in cls_indices]

            cls_bboxes = [
                [
                    boxes[i][0] / img_size[0],
                    boxes[i][1] / img_size[1],
                    boxes[i][2] / img_size[0],
                    boxes[i][3] / img_size[1],
                ]
                for i in cls_indices
            ]
            cls_confs = [
                boxes[i][4] for i in cls_indices
            ]  # Extract the confidence values
            if cls_bboxes:
                mean_bbox = np.mean(cls_bboxes, axis=0).tolist()
                std_bbox = np.std(cls_bboxes, axis=0).tolist()
                mean_conf = np.mean(cls_confs).tolist()

                bbox_features.extend(mean_bbox + std_bbox + [mean_conf])
            else:
                bbox_features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])

        features = class_features + bbox_features

        return features

    @staticmethod
    def preprocess_inference_data_onehot(
        classes: List[str],
        unique_classes: List[str],
    ) -> List[int]:
        return [int(classes.count(cls) > 0) for cls in unique_classes]
