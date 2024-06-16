import yaml
import torch
import numpy as np

from collections import OrderedDict
from typing import List, Optional, Tuple, Union
from src.image_classifiers.classifiers.binary.binary_classifier import BinaryClassifier
from src.image_classifiers.classifiers.multilabel.multilabel_classifier import (
    MultilabelClassifier,
)
from src.image_classifiers.classifiers.singlelabel.singlelabel_classifier import (
    SinglelabelClassifier,
)


class BaseImageClassifier:
    def __init__(self) -> None:
        self.device = "cpu"

    def setup_classifier(
        self, model_path: str, cfg_path: str, classifier_type: str
    ) -> Tuple[
        Union[SinglelabelClassifier, MultilabelClassifier, BinaryClassifier],
        np.array,
        Optional[np.array],
    ]:
        with open(cfg_path, "r") as file:
            try:
                args = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                raise ValueError(f"Error loading YAML {cfg_path}: {exc}") from exc

        classes = args["classes"]
        class_weights = args["class_weights"]

        if classifier_type == "single":
            model = SinglelabelClassifier(args)
        elif classifier_type == "multi":
            model = MultilabelClassifier(args)
        elif classifier_type == "binary":
            model = BinaryClassifier(args)

        saved_state_dict = torch.load(
            model_path, map_location=torch.device(self.device)
        )
        new_state_dict = OrderedDict()

        for k, v in saved_state_dict.items():
            name = k if k.startswith("base_model.") else f"base_model.{k}"
            new_state_dict[name] = v

        new_state_dict2 = OrderedDict()

        for k, v in new_state_dict.items():
            name = (
                k.replace("base_model.", "")
                if k.startswith("base_model.last_layer")
                else k
            )
            new_state_dict2[name] = v

        model.load_state_dict(new_state_dict2)
        model.to(self.device)
        model.eval()

        return model, np.array(classes), np.array(class_weights)

    def make_prediction(
        self,
        input_data: torch.Tensor,
        model: MultilabelClassifier,
    ) -> np.ndarray:
        with torch.no_grad():
            output = model(input_data).cpu()
            output = np.array(output)
            predictions = (output >= 0.5).astype(int)
        return predictions

    def make_prediction_bin(
        self,
        input_data: torch.Tensor,
        model: BinaryClassifier,
    ) -> int:
        with torch.no_grad():
            pred = model(input_data).cpu()
            pred = 1 if pred > 0.5 else 0
        return pred

    def make_prediction_one_cls(
        self,
        input_data: torch.Tensor,
        model: SinglelabelClassifier,
    ) -> int:
        with torch.no_grad():
            output = model(input_data).cpu()
            output = np.array(output)
            prediction = np.argmax(output)
        return prediction
