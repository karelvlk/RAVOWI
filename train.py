import argparse
import logging
from typing import Dict, List, Union

import torch

from src.image_classifiers.classifiers.binary.binary_trainer import BinaryTrainer
from src.image_classifiers.classifiers.multilabel.multilabel_trainer import (
    MultilabelTrainer,
)
from src.image_classifiers.classifiers.singlelabel.singlelabel_trainer import (
    SinglelabelTrainer,
)
from src.object_detection.trainers.yolo_trainer import YOLO_trainer


def train_yolo(
    name: str,
    dataset_path: str,
    weights_path: str,
    override: Dict[str, Union[int, float, str]],
) -> None:
    trainer = YOLO_trainer(
        name,
        dataset_path,
        weights_path,
        override,
    )
    trainer()


def train_classification(
    cls_type: str,
    data_json_path: str,
    dataloader_type: str,
    cfg: str,
    name: str,
    classes: List[str],
) -> None:
    torch.cuda.empty_cache()

    name = name or "no_name"

    match cls_type:
        case "binary":
            clt = BinaryTrainer(name, classes, cfg, data_json_path, dataloader_type)
        case "single":
            clt = SinglelabelTrainer(
                name, classes, cfg, data_json_path, dataloader_type
            )
        case "multi":
            clt = MultilabelTrainer(name, classes, cfg, data_json_path, dataloader_type)

    clt.train()

def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mode", choices=["yolo", "cls", "blur"], required=True, help="Operating mode"
    )

    args, _ = parser.parse_known_args()

    match args.mode:
        case "yolo":
            parser.add_argument("--name", type=str, help="Name of trained model")
            parser.add_argument(
                "--dataset_path",
                type=str,
                required=False,
                help="Path to dataset or bucket name",
            )
            parser.add_argument(
                "--weights_path",
                type=str,
                required=False,
                help="Path to weights or bucket name",
            )
            parser.add_argument(
                "--override",
                nargs="*",
                required=False,
                help="Additional key-value pairs (delimiter is '=') to be overridden in train config",
            )

            args = parser.parse_args()

            # Processing the override argument
            override: Dict[str, Union[int, float, str]] = {}
            if args.override:
                for arg in args.override:
                    key, value = arg.split("=")
                    if value.isdigit():
                        value = int(value)
                    elif value.replace(".", "", 1).isdigit() and value.count(".") < 2:
                        value = float(value)
                    override[key] = value

            train_yolo(
                args.name,
                args.dataset_path,
                args.weights_path,
                override,
            )
        case "cls":
            parser.add_argument(
                "--type",
                choices=["binary", "single", "multi"],
                required=True,
                help="Type of training: binary, single, multi",
            )
            parser.add_argument(
                "--data", type=str, required=True, help="Path to data json"
            )
            parser.add_argument(
                "--cfg", type=str, required=True, help="Path to cfg yaml"
            )
            parser.add_argument("--name", type=str, required=True, help="name")
            parser.add_argument(
                "--dataloader",
                choices=["json", "image"],
                required=True,
                help="Type of dataloader json/image",
            )
            parser.add_argument(
                "--classes", nargs="+", type=str, required=True, help="Array of classes"
            )
            args = parser.parse_args()

            train_classification(
                args.type, args.data, args.dataloader, args.cfg, args.name, args.classes
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s-%(name)s-%(levelname)s:%(message)s",
    )
    main()
