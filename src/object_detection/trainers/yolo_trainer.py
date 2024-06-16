import logging
import os
import re
import subprocess
import time
from typing import Any, Dict, Union

import torch
from ultralytics import YOLO

from src.data_preprocessors.yolo_data_preprocessor import YOLODataPreprocessor
from src.profiler import Profiler
from src.utils import yaml_load


class YOLO_trainer:
    def __init__(
        self,
        name: str,
        dataset_path: str,
        is_dataset_local: bool,
        weights_path: str,
        is_weights_local: bool,
        weights_uploader_bucket_name: str,
        weights_uploader_upload_dir_name: str,
        override: Dict[str, Union[str, int, float]] = {},
    ) -> None:
        self.name = name
        self.is_dataset_local = is_dataset_local
        self.is_weights_local = is_weights_local
        self.override = override

        self.dataset_path = "datasets/" + dataset_path.split("/", 1)[1]
        self.pretrained_yolo_path = weights_path

        self.yolo_preprocessor = YOLODataPreprocessor(self.dataset_path)

    def __call__(self) -> None:
        self.preprocess()
        cfg = self.prepare_cfg(
            cfg_path=os.path.join(self.dataset_path, "cfg.yaml"),
            pretrained_yolo_path=self.pretrained_yolo_path,
            data_path=self.dataset_path,
            name=self.name,
        )
        logging.info(f"Yolo training config: {cfg}")
        model, _ = self.train(
            cfg=cfg,
            yolo_path=self.pretrained_yolo_path,
        )
        self.postprocess(model)

    def prepare_cfg(
        self, cfg_path: str, pretrained_yolo_path: str, data_path: str, name: str
    ) -> Dict[str, Any]:
        cfg = yaml_load(cfg_path)
        cfg["device"] = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        if data_path:
            cfg["data"] = os.path.join(os.getcwd(), data_path, "data.yaml")

        if name:
            cfg["name"] = name

        if pretrained_yolo_path:
            cfg["model"] = pretrained_yolo_path

        logging.info(f"Default training params are overrided by {self.override}.")
        cfg.update(self.override)

        return cfg

    def preprocess(self) -> None:
        with Profiler("Dataset dowload", logging):
            self.download_dataset()

        with Profiler("Preprocess dataset", logging):
            self.yolo_preprocessor.preprocess()

    def postprocess(self, model: YOLO) -> None:
        metrics = model.val()
        metrics_dict = metrics.results_dict
        mAP50 = metrics_dict["metrics/mAP50(B)"]
        mAP5095 = metrics_dict["metrics/mAP50-95(B)"]
        logging.info(
            f"Model training final metrics: {metrics_dict} \n mAP50: {mAP50} \n mAP50-95: {mAP5095}"
        )
        export_path = model.export()
        base_dir = os.path.dirname(os.path.dirname(export_path))
        logging.info(f"Train base_dir is: {base_dir}")
        logging.info(f"Model exported to {export_path}")

    def train(
        self,
        cfg: Dict[str, Any],
        yolo_path: str = None,
    ) -> YOLO:
        torch.cuda.empty_cache()
        model = YOLO(yolo_path, task="detect")
        model.load(cfg["model"])
        model.train(**cfg)
        logging.info("Model training finished")
        return model, cfg
