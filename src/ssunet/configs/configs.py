"""Configuration for the project."""

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy

import pytorch_lightning as pl
import torch

from ..constants import DEFAULT_CONFIG_PATH
from ..utils import _load_yaml
from .data_config import DataConfig
from .file_config import PathConfig, SplitParams
from .model_config import ModelConfig
from .train_config import LoaderConfig, TrainConfig


@dataclass
class MasterConfig:
    """Configuration class containing all configurations."""

    data_config: DataConfig
    path_config: PathConfig
    split_params: SplitParams
    model_config: ModelConfig
    loader_config: LoaderConfig
    train_config: TrainConfig

    target_path: Path = field(init=False)
    time_stamp: str = field(init=False)

    def __post_init__(self):
        """Post initialization function."""
        self.time_stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    @property
    def name(self) -> str:
        """Generate a name for the experiment."""
        name = "_".join(
            [
                self.time_stamp,
                self.train_config.name,
                self.data_config.name,
                self.model_config.name,
                # self.loader_config.name,
            ]
        )
        return name

    @property
    def device(self) -> torch.device:
        """Use the configured accelerator for the inference phase as well as training."""
        accelerator = self.train_config.accelerator
        if accelerator == "cpu" or (accelerator == "auto" and not torch.cuda.is_available()):
            return torch.device("cpu")
        if accelerator == "mps":
            return torch.device("mps")
        devices = self.train_config.devices
        index = devices[0] if isinstance(devices, (list, tuple)) else 0
        return torch.device(f"cuda:{index}")

    @property
    def data_path(self) -> Path:
        """Get the path to the data."""
        return self.path_config.data_file

    @property
    def model_path(self) -> Path:
        """Get the path to the model."""
        return self.train_config.default_root_dir

    @property
    def log_path(self) -> Path:
        """Get the path to the log."""
        return self.train_config.default_root_dir / "logs"

    @property
    def checkpoint_path(self) -> Path:
        """Get the path to the checkpoint."""
        return self.train_config.default_root_dir / "model.ckpt"

    @property
    def trainer(self) -> pl.Trainer:
        """Alias for the trainer."""
        return self.train_config.trainer

    def copy_config(self, source_path: Path | str, target_path: Path | str | None = None) -> None:
        """Copy the configuration to a new directory."""
        source_path = Path(source_path)
        source_file_name = source_path.name
        target_path = Path(target_path) if target_path is not None else self.target_path
        target_path.mkdir(parents=True, exist_ok=True)
        destination = target_path / source_file_name
        if source_path.resolve() != destination.resolve():
            copy(source_path, destination)

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "MasterConfig":
        """Convert the configuration dictionary to dataclasses."""
        config_path = Path(config_path)
        config = _load_yaml(config_path)
        master_config = MasterConfig(
            path_config=PathConfig(**config["PATH"]),
            data_config=DataConfig(**config["DATA"]),
            split_params=SplitParams(**config["SPLIT"]),
            model_config=ModelConfig(**config["MODEL"]),
            loader_config=LoaderConfig(**config["LOADER"]),
            train_config=TrainConfig(**config["TRAIN"]),
        )
        if re.match(r"^\d{8}_\d{6}_", config_path.parent.name):
            # set the root to the parent directory of the config file
            master_config.time_stamp = config_path.parent.name[:15]
            master_config.train_config.set_new_root(new_root=config_path.parent)
            master_config.target_path = config_path.parent
        else:
            # set the root to the name of the experiment
            experiment_name = master_config.name
            candidate = master_config.train_config.default_root_dir / experiment_name
            # TensorBoard's event writer uses paths subject to Windows MAX_PATH.
            # Leave room for its log directories and generated event filename.
            if os.name == "nt" and len(str(candidate.absolute())) > 160:
                digest = hashlib.sha256(experiment_name.encode()).hexdigest()[:12]
                experiment_name = f"{master_config.time_stamp}_{digest}"
            master_config.train_config.set_new_root(new_root=experiment_name)
            master_config.target_path = master_config.train_config.default_root_dir
        return master_config
