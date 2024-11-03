"""Training script."""

import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
import torch.utils.data as dt
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    Callback,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from ..constants import LOGGER


@dataclass
class LoaderConfig:
    """Data loader configuration."""

    batch_size: int = 20
    shuffle: bool = False
    pin_memory: bool = False
    drop_last: bool = True
    num_workers: int = 6
    persistent_workers: bool = True
    prefetch_factor: int | None = None

    @property
    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @property
    def name(self) -> str:
        """Get the name of the dataloader."""
        name_str = [
            f"bs={self.batch_size}",
            f"sh={self.shuffle}",
            f"pm={self.pin_memory}",
            f"dl={self.drop_last}",
            f"nw={self.num_workers}",
            f"pw={self.persistent_workers}",
        ]
        return "_".join(name for name in name_str if name is not None and name != "")

    def loader(self, data: dt.Dataset) -> dt.DataLoader:
        """Create a data loader with CUDA stream support.

        :param data: Dataset to load
        :type data: dt.Dataset
        :return: DataLoader configured with CUDA streaming
        :rtype: dt.DataLoader
        """
        return dt.DataLoader(data, **self.to_dict)


@dataclass
class TrainConfig:
    """Training configuration."""

    default_root_dir: str | Path = Path("../models")
    accelerator: str = "cuda"
    gradient_clip_val: float = 1.0
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
    callbacks_save_on_train_end: bool = False
    callbacks_handle_interrupt: bool = True

    # device stats monitor
    callbacks_device_stats_monitor: bool = False

    # other params
    logger_name: str = "logs"
    profiler: str = "simple"
    limit_val_batches: int = 20
    log_every_n_steps: int = 20
    note: str = ""

    matmul_precision: Literal["highest", "high", "medium"] = "high"

    def __post_init__(self):
        """Setting the model root directory and matmul precision."""
        self.default_root_dir = Path(self.default_root_dir)
        torch.set_float32_matmul_precision(self.matmul_precision)
        self.default_root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        """Get the name of the training session."""
        name_parts = [
            f"e={self.max_epochs}",
            f"p={self.precision}",
            f"d={self.devices}",
        ]
        if self.note:
            name_parts.append(f"n={self.note}")
        return "_".join(name_parts)

    @property
    def model_checkpoint(self) -> ModelCheckpoint:
        """Create a model checkpoint callback."""
        return ModelCheckpoint(
            save_weights_only=self.mc_save_weights_only,
            mode=self.mc_mode,
            monitor=self.mc_monitor,
            save_top_k=self.mc_save_top_k,
        )

    @property
    def learning_rate_monitor(self) -> LearningRateMonitor:
        """Create a learning rate monitor callback."""
        return LearningRateMonitor(self.lrm_logging_interval)

    @property
    def early_stopping(self) -> EarlyStopping:
        """Create an early stopping callback."""
        return EarlyStopping(self.es_monitor, patience=self.es_patience)

    @property
    def logger(self) -> TensorBoardLogger:
        """Create a logger."""
        logger_path = Path(self.default_root_dir) / self.logger_name
        if not logger_path.exists():
            logger_path.mkdir(parents=True, exist_ok=True)
        return TensorBoardLogger(save_dir=self.default_root_dir, name=self.logger_name)

    @property
    def callbacks(self) -> list:
        """Create a list of callbacks."""
        callbacks = []
        if self.callbacks_model_checkpoint:
            callbacks.append(self.model_checkpoint)
        if self.callbacks_learning_rate_monitor:
            callbacks.append(self.learning_rate_monitor)
        if self.callbacks_early_stopping:
            callbacks.append(self.early_stopping)
        if self.callbacks_device_stats_monitor:
            callbacks.append(DeviceStatsMonitor())
        if self.callbacks_save_on_train_end:
            callbacks.append(self.model_save_on_train_end)
        if self.callbacks_handle_interrupt:
            callbacks.append(self.interrupt_callback)
        return callbacks

    @property
    def model_save_on_train_end(self) -> Callback:
        """Create a callback to save model on training end."""

        class SaveOnTrainEnd(Callback):
            def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
                trainer.save_checkpoint(Path(trainer.default_root_dir) / "final_model.ckpt")
                LOGGER.info(
                    "Model saved as %s.", Path(trainer.default_root_dir) / "final_model.ckpt"
                )

        return SaveOnTrainEnd()

    @property
    def interrupt_callback(self) -> Callback:
        """Create a callback to handle training interruption."""

        class HandleInterrupt(Callback):
            def setup(
                self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str | None = None
            ) -> None:
                """Set up custom interrupt handler."""

                def handle_interrupt(signum, frame):
                    """Custom interrupt handler."""
                    save_path = Path(trainer.default_root_dir) / "interrupted_model.ckpt"
                    try:
                        trainer.save_checkpoint(save_path)
                        LOGGER.info("Training interrupted. Model saved to %s", save_path)
                    except FileNotFoundError as err:
                        LOGGER.error("Failed to save model on interrupt: %s", str(err))
                    finally:
                        sys.exit(0)

                # Register our custom handler
                signal.signal(signal.SIGINT, handle_interrupt)

        return HandleInterrupt()

    @property
    def trainer(self) -> pl.Trainer:
        """Create a trainer."""
        return pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            precision=self.precision,
            max_epochs=self.max_epochs,
            logger=self.logger,
            callbacks=self.callbacks,
            default_root_dir=self.default_root_dir,
            gradient_clip_val=self.gradient_clip_val,
            limit_val_batches=self.limit_val_batches,
            log_every_n_steps=self.log_every_n_steps,
            profiler=self.profiler,
            enable_checkpointing=self.callbacks_model_checkpoint,
            enable_progress_bar=True,
            use_distributed_sampler=True,
            detect_anomaly=False,
        )

    @property
    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def set_new_root(self, new_root: Path | str) -> None:
        """Set a new model root directory.

        :param new_root: New root directory path. If a string, will be joined to existing root dir
        :type new_root: Path | str
        """
        self.default_root_dir = (
            Path(self.default_root_dir) / new_root if isinstance(new_root, str) else new_root
        )
