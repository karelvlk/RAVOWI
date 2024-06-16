import abc
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision
from torch import nn


class BaseClassificationModel(nn.Module, abc.ABC):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.args = args
        self.base_model, self.last_layer = self.build_model()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward function is not yet implemented.")

    def build_layer(self, layer_cfg: Dict[str, Any], i: int) -> Tuple[str, nn.Module]:
        layer_type = getattr(nn, layer_cfg["type"])
        layer_args = {k: v for k, v in layer_cfg.items() if k not in ["type", "name"]}
        layer_name = layer_cfg["name"] if "name" in layer_cfg else f"nonename{i}"
        return layer_name, layer_type(**layer_args)

    def build_custom_model(self) -> Tuple[nn.Module, Optional[nn.Module]]:
        custom_model_cfg = self.args["model"]["custom_model"]
        model_name = custom_model_cfg["name"]
        layers_cfg = custom_model_cfg["layers"]

        layers = OrderedDict()
        for i, layer_cfg in enumerate(layers_cfg):
            layer_name, layer = self.build_layer(layer_cfg, i)
            layers[layer_name] = layer

        custom_model = nn.Sequential(layers)

        return custom_model, None

    def build_model(self) -> Tuple[nn.Module, Optional[nn.Module]]:
        if "custom_model" in self.args["model"]:
            return self.build_custom_model()

        base_model = getattr(
            torchvision.models, self.args["model"]["base_model"]["name"]
        )
        base_model = base_model(weights=self.args["model"]["base_model"]["weights"])
        last_layer_name = self.args["model"]["base_model"]["last_layer"]
        last_layer = getattr(base_model, last_layer_name)
        specific_layer_index = self.args["model"]["base_model"]["specific_layer_index"]

        if specific_layer_index is not None:
            specific_layer = last_layer[specific_layer_index]
            num_ftrs = specific_layer.in_features
        else:
            num_ftrs = last_layer.in_features

        classifier_layers: List[nn.Module] = []
        for i, layer_cfg in enumerate(self.args["model"]["classifier"]["layers"]):
            # Check if in_features is in the configuration and replace it with the correct value
            for key in layer_cfg:
                if layer_cfg[key] == "num_ftrs":
                    layer_cfg[key] = num_ftrs

            _, layer = self.build_layer(layer_cfg, i)
            classifier_layers.append(layer)
            # Update num_ftrs for the next layer_cfg if it has in_features
            if "out_features" in layer_cfg:
                num_ftrs = layer_cfg["out_features"]

        finetune_head = nn.Sequential(*classifier_layers)

        if specific_layer_index is not None:
            new_classifier = list(base_model.classifier)
            new_classifier[specific_layer_index] = finetune_head
            setattr(base_model, last_layer_name, nn.Sequential(*new_classifier))
        else:
            setattr(base_model, last_layer_name, finetune_head)

        return base_model, finetune_head
