# type: ignore

import abc
import datetime
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import yaml
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from src.image_classifiers.classifiers.metrics import categorical_accuracy, macro_f1
from src.image_classifiers.datasets.binary_image_dataset_loader import (
    BinaryImageDatasetLoader,
)
from src.image_classifiers.datasets.json_dataset_loader import JsonDatasetLoader
from src.image_classifiers.datasets.multi_image_dataset_loader import (
    MultiImageDatasetLoader,
)


class BaseClassificationTrainer(abc.ABC):
    def __init__(
        self,
        name: str,
        classes: List[str],
        cfg_path: Optional[str],
        data_path: str,
        model: Optional[nn.Module] = None,
        train: bool = True,
    ) -> None:
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = name
        self.classes = classes
        self.log_dir = f"cls2/classification_{self.name}_{self.timestamp}_log"

        self.args = self.get_args(cfg_path)
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.data_path = data_path
        self.weights = (
            self.args["class_weights"] if "class_weights" in self.args else None
        )

        # Create a custom logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create handlers
        if train:
            os.makedirs(self.log_dir, exist_ok=True)

            log_filename = f"{self.log_dir}/log.txt"

            c_handler = logging.StreamHandler()  # console handler
            f_handler = logging.FileHandler(log_filename)  # file handler

            c_handler.setLevel(logging.INFO)
            f_handler.setLevel(logging.INFO)

            # Create formatters and add it to handlers
            c_format = logging.Formatter("[%(asctime)s]: %(message)s")
            f_format = logging.Formatter("[%(asctime)s]: %(message)s")

            c_handler.setFormatter(c_format)
            f_handler.setFormatter(f_format)

            # Add handlers to the logger
            self.logger.addHandler(c_handler)
            self.logger.addHandler(f_handler)

            self.dataloaders, self.dataset_sizes = self.get_dataloaders(data_path)
            self.model = self.get_model()

    def get_args(
        self, cfg_path: Optional[str]
    ) -> Dict[str, Union[str, Dict[str, Union[str, int]]]]:
        if cfg_path is None:
            cfg_path = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cfg.yaml")
            )

        with open(cfg_path, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "cfg.yaml"), "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
        return data

    @abc.abstractmethod
    def get_model(self) -> nn.Module:
        raise NotImplementedError("Get model is not implemented in base trainer")

    @abc.abstractmethod
    def get_loss_fn(self, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        raise NotImplementedError(
            "Get loss function is not implemented in base trainer"
        )

    @abc.abstractmethod
    def get_dataloader(
        self, num_classes: int, data_path: str, log_dir: str
    ) -> Union[MultiImageDatasetLoader, BinaryImageDatasetLoader, JsonDatasetLoader]:
        raise NotImplementedError("Get dataloader is not implemented in base trainer")

    def build_scheduler(
        self, type: str, optimizer: optim.Optimizer, params_dict: Dict[str, int]
    ) -> lr_scheduler:
        scheduler_cfg = self.args[f"scheduler_{type}"]
        scheduler_type = getattr(lr_scheduler, scheduler_cfg["type"])
        # scheduler_params = {k: eval(v, {"__builtins__": None}, params_dict) for k, v in scheduler_cfg['params'].items()}
        scheduler_params = {
            k: eval(str(v), {"__builtins__": None}, params_dict)
            for k, v in scheduler_cfg["params"].items()
        }

        return scheduler_type(optimizer, **scheduler_params)

    def build_optimizer(self, opt_type: str, model: nn.Module) -> optim.Optimizer:
        optim_cfg = self.args[f"optim_{opt_type}"]
        optim_name = optim_cfg["type"]
        optim_args = optim_cfg.get("params", {})
        logging.debug(f"optim_args {optim_args}")
        optimizer_constructor = getattr(optim, optim_name)
        optimizer = optimizer_constructor(
            filter(lambda p: p.requires_grad, model.parameters()), **optim_args
        )
        return optimizer

    def is_finetuning_model(self) -> bool:
        return "custom_model" not in self.args["model"]

    def get_head_scheduler(self) -> None:
        raise NotImplementedError

    def get_whole_scheduler(self) -> None:
        raise NotImplementedError

    def get_dataloaders(
        self, data_json_path: str
    ) -> Tuple[Dict[str, torch.utils.data.DataLoader], Dict[str, int]]:
        data_loader = self.get_dataloader(
            len(self.classes), self.data_path, self.log_dir
        )
        train_dataloader, val_dataloader, test_dataloader = data_loader()

        dataloaders = {"train": train_dataloader, "val": val_dataloader}
        dataset_sizes = {
            "train": len(train_dataloader.dataset),
            "val": len(val_dataloader.dataset),
        }

        logging.debug(f"dataset_sizes: {dataset_sizes}")
        return dataloaders, dataset_sizes

    def get_metrics(
        self, pred: torch.Tensor, gold: torch.Tensor
    ) -> Tuple[float, float, float]:
        # pred = pred.cpu()  # move to cpu before converting to numpy
        # gold = gold.cpu()  # move to cpu before converting to numpy

        # accu = categorical_accuracy(pred, gold)
        # f1 = macro_f1(pred, gold)

        return None, None, None

    def train(self) -> None:
        if self.is_finetuning_model():
            self.train_2_segment()
        else:
            self.train_1_segment()

    def train_1_segment(self) -> None:
        model = self.model.to(self.device)
        grad_clip = self.args["grad_clip"]
        criterion = self.get_loss_fn()
        optimizer = self.build_optimizer("custom", model)
        epochs = self.args["epochs_custom"]
        params_dict = {
            "epochs": epochs,
            "data_length": len(self.dataloaders["train"]),
        }
        scheduler = self.build_scheduler("custom", optimizer, params_dict)

        self.train_model(
            "CUSTOM", model, criterion, optimizer, scheduler, grad_clip, epochs
        )

    def train_2_segment(self) -> None:
        model = self.model.to(self.device)
        grad_clip = self.args["grad_clip"]
        class_weights = (
            None if self.weights is None else torch.tensor(self.weights).to(self.device)
        )
        criterion = self.get_loss_fn(class_weights=class_weights)

        # ---------------
        # Part 1: Train the head
        # ---------------
        for param in model.parameters():
            param.requires_grad = False

        for param in model.last_layer.parameters():
            param.requires_grad = True

        optimizer = self.build_optimizer("head", model)
        head_epochs = self.args["epochs_head"]

        params_dict = {
            "epochs": head_epochs,
            "data_length": len(self.dataloaders["train"]),
        }
        scheduler = self.build_scheduler("head", optimizer, params_dict)
        self.train_model(
            "HEAD", model, criterion, optimizer, scheduler, grad_clip, head_epochs
        )
        torch.cuda.empty_cache()

        # ---------------
        # Part 2: Train the whole model
        # ---------------
        for param in model.parameters():
            param.requires_grad = True

        optimizer = self.build_optimizer("whole", model)
        whole_epochs = self.args["epochs_whole"]
        params_dict = {
            "epochs": whole_epochs,
            "data_length": len(self.dataloaders["train"]),
        }
        scheduler = self.build_scheduler("whole", optimizer, params_dict)
        self.train_model(
            "WHOLE", model, criterion, optimizer, scheduler, grad_clip, whole_epochs
        )

    def train_model(
        self,
        name: str,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: lr_scheduler,
        grad_clip: float,
        num_epochs: int,
    ) -> None:
        loss_stats = {"train": [], "val": []}
        accu_stats = {"train": [], "val": []}
        f1_stats = {"train": [], "val": []}

        self.logger.info(f"[START --- {name} ---]")
        best_acc = 0.0
        for epoch in tqdm(range(1, num_epochs + 1)):
            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    train_epoch_loss = 0
                    train_epoch_acc = 0
                    train_epoch_f1 = 0

                    model.train()  # Set model to training mode
                    for x_train, y_train in tqdm(self.dataloaders[phase]):
                        if (
                            x_train is None
                            or y_train is None
                            or len(x_train) <= 1
                            or len(y_train) <= 1
                        ):
                            logging.debug(f"x_train {x_train}; y_train {y_train}")
                            continue

                        x_train = x_train.to(
                            "cuda"
                            if torch.cuda.is_available()
                            else ("mps" if torch.backends.mps.is_available() else "cpu")
                        )
                        y_train = y_train.to(
                            "cuda"
                            if torch.cuda.is_available()
                            else ("mps" if torch.backends.mps.is_available() else "cpu")
                        )

                        y_train_pred = model(x_train).squeeze()
                        train_loss = criterion(y_train_pred, y_train)
                        acc_train, f1_train, _ = self.get_metrics(y_train_pred, y_train)

                        # Gradient clipping
                        if grad_clip:
                            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

                        train_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        # (1/len(self.dataloaders[phase]))
                        train_epoch_loss += train_loss.item() * x_train.size(0)
                        # (1/len(self.dataloaders[phase]))
                        train_epoch_acc += acc_train * x_train.size(0)
                        # (1/len(self.dataloaders[phase]))
                        train_epoch_f1 += f1_train * x_train.size(0)

                    scheduler.step()
                    loss_stats[phase].append(
                        train_epoch_loss / len(self.dataloaders[phase].dataset)
                    )
                    accu_stats[phase].append(
                        train_epoch_acc / len(self.dataloaders[phase].dataset)
                    )
                    f1_stats[phase].append(
                        train_epoch_f1 / len(self.dataloaders[phase].dataset)
                    )

                elif phase == "val":
                    val_epoch_loss = 0
                    val_epoch_acc = 0
                    val_epoch_f1 = 0

                    with torch.no_grad():
                        model.eval()  # Set model to evaluate mode

                        for x_val, y_val in self.dataloaders[phase]:
                            if (
                                x_val is None
                                or y_val is None
                                or len(x_val) <= 1
                                or len(y_val) <= 1
                            ):
                                logging.debug(f"train > x_val {x_val}; y_val {y_val}")
                                continue

                            x_val = x_val.to(
                                "cuda"
                                if torch.cuda.is_available()
                                else (
                                    "mps"
                                    if torch.backends.mps.is_available()
                                    else "cpu"
                                )
                            )
                            y_val = y_val.to(
                                "cuda"
                                if torch.cuda.is_available()
                                else (
                                    "mps"
                                    if torch.backends.mps.is_available()
                                    else "cpu"
                                )
                            )

                            y_val_pred = model(x_val).squeeze()

                            val_loss = criterion(y_val_pred, y_val)
                            acc_val, f1_val, _ = self.get_metrics(y_val_pred, y_val)

                            # (1/len(self.dataloaders[phase]))
                            val_epoch_loss += val_loss.item() * x_val.size(0)
                            # (1/len(self.dataloaders[phase]))
                            val_epoch_acc += acc_val * x_val.size(0)
                            # (1/len(self.dataloaders[phase]))
                            val_epoch_f1 += f1_val * x_val.size(0)

                    loss_stats[phase].append(
                        val_epoch_loss / len(self.dataloaders[phase].dataset)
                    )
                    accu_stats[phase].append(
                        val_epoch_acc / len(self.dataloaders[phase].dataset)
                    )
                    f1_stats[phase].append(
                        val_epoch_f1 / len(self.dataloaders[phase].dataset)
                    )

                    if f1_stats["val"][-1] > best_acc:
                        best_acc = f1_stats["val"][-1]
                        torch.save(model.state_dict(), f"{self.log_dir}/best.pt")

            log = f'[{name}] Epoch {epoch+0:02}: | Train Loss: {loss_stats["train"][-1]:.5f} | Val Loss: {loss_stats["val"][-1]:.5f} | Train Acc (%): {accu_stats["train"][-1]:.3f}| Val Acc (%): {accu_stats["val"][-1]:.3f} | Train F1: {f1_stats["train"][-1]:.3f}| Val F1: {f1_stats["val"][-1]:.3f}'
            self.logger.info(log)

            torch.save(model.state_dict(), f"{self.log_dir}/last.pt")

        self.logger.info(f"[{name}]: Best val Acc: {best_acc:4f}")
        self.logger.info(f"[END --- {name} ---]")
        logging.info(f"[{name}]: Training finished - Best val Acc: {best_acc:4f}")
