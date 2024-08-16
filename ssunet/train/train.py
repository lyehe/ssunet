import torch
import pytorch_lightning as pl

from datetime import datetime
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)

from dataclasses import dataclass, field
from typing import Literal
from logging import getLogger

logger = getLogger(__name__)


@dataclass
class LoaderConfig:
    batch_size: int = 20
    shuffle: bool = False
    pin_memory: bool = False
    drop_last: bool = True
    num_workers: int = 6
    persistent_workers: bool = True

    @property
    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
        }

    @property
    def name(self) -> str:
        name_str = [
            f"bs={self.batch_size}",
            f"sh={self.shuffle}",
            f"pm={self.pin_memory}",
            f"dl={self.drop_last}",
            f"nw={self.num_workers}",
            f"pw={self.persistent_workers}",
        ]
        return "_".join(name for name in name_str if name is not None and name != "")

    def __call__(self) -> dict:
        return self.to_dict


@dataclass
class TrainConfig:
    default_root_dir: str = "./models"
    accelerator: str = "cuda"
    gradient_clip_val: int = 1
    precision: str | int | None = 32
    max_epochs: int = 50
    devices: int | list[int] = 0

    # callbacks - model checkpoint
    callbacks_model_checkpoint: bool = True
    mc_save_weights_only: bool = True
    mc_mode: str = "min"
    mc_monitor: str = "val_loss"
    mc_save_top_k: int = 2

    # learning rate monitor
    callbacks_learning_rate_monitor: bool = True
    lrm_logging_interval: Literal["step", "epoch"] | None = "epoch"

    # early stopping
    callbacks_early_stopping: bool = False
    es_monitor: str = "val_loss"
    es_patience: int = 25

    # device stats monitor
    callbacks_device_stats_monitor: bool = False

    # other params
    logger_name: str = "logs"
    profiler: str = "simple"
    limit_val_batches: int = 20
    log_every_n_steps: int = 20
    note: str = ""

    matmul_precision: str = "high"
    time_stamp: str = field(init=False)

    def __post_init__(self):
        torch.set_float32_matmul_precision(self.matmul_precision)
        self.time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def name(self) -> str:
        name_str = [
            f"{self.time_stamp}",
            f"e={self.max_epochs}",
            f"p={self.precision}",
            f"n={self.note}" if self.note != "" else None,
        ]
        return "_".join(name for name in name_str if name is not None and name != "")

    @property
    def model_checkpoint(self) -> ModelCheckpoint:
        return ModelCheckpoint(
            save_weights_only=self.mc_save_weights_only,
            mode=self.mc_mode,
            monitor=self.mc_monitor,
            save_top_k=self.mc_save_top_k,
        )

    @property
    def learning_rate_monitor(self) -> LearningRateMonitor:
        return LearningRateMonitor(self.lrm_logging_interval)

    @property
    def early_stopping(self) -> EarlyStopping:
        return EarlyStopping(self.es_monitor, patience=self.es_patience)

    @property
    def logger(self) -> TensorBoardLogger:
        return TensorBoardLogger(save_dir=self.default_root_dir, name=self.logger_name)

    @property
    def callbacks(self) -> list:
        output = []
        if self.callbacks_model_checkpoint:
            output.append(self.model_checkpoint)
        if self.callbacks_learning_rate_monitor:
            output.append(self.learning_rate_monitor)
        if self.callbacks_early_stopping:
            output.append(self.early_stopping)
        if self.callbacks_device_stats_monitor:
            output.append(DeviceStatsMonitor())
        return output

    @property
    def to_dict(self) -> dict:
        return {
            "default_root_dir": self.default_root_dir,
            "accelerator": self.accelerator,
            "gradient_clip_val": self.gradient_clip_val,
            "precision": self.precision,
            "max_epochs": self.max_epochs,
            "devices": self.devices,
            "callbacks": self.callbacks,
            "logger": self.logger,
            "profiler": self.profiler,
            "limit_val_batches": self.limit_val_batches,
            "log_every_n_steps": self.log_every_n_steps,
        }

    @property
    def trainer(self) -> pl.Trainer:
        """Create  a PyTorch Lightning Trainer instance from the config

        :return: The PyTorch Lightning Trainer
        :rtype: pl.Trainer
        """
        return pl.Trainer(**self.to_dict)
