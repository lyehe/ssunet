"""Configuration for the project."""

from dataclasses import dataclass
from pathlib import Path
from shutil import copy

import pytorch_lightning as pl

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
    
    @property
    def name(self) -> str:
        """Generate a name for the experiment."""
        name = "_".join(
            [
                self.data_config.name,
                self.model_config.name,
                # self.loader_config.name,
                # self.train_config.name,
            ]
        )
        return name

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
    def config_path(self) -> Path:
        """Get the path to the config."""
        return self.train_config.default_root_dir / "config.yml"

    @property
    def trainer(self) -> pl.Trainer:
        """Alias for the trainer."""
        return self.train_config.trainer


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> MasterConfig:
    """Convert the configuration dictionary to dataclasses."""
    config = _load_yaml(config_path)
    master_config = MasterConfig(
        path_config=PathConfig(**config["PATH"]),
        data_config=DataConfig(**config["DATA"]),
        split_params=SplitParams(**config["SPLIT"]),
        model_config=ModelConfig(**config["MODEL"]),
        loader_config=LoaderConfig(**config["LOADER"]),
        train_config=TrainConfig(**config["TRAIN"]),
    )
    master_config.train_config.set_new_root(new_root=master_config.name)
    return master_config


def copy_config(source_path: Path | str, target_path: Path | str) -> None:
    """Copy the configuration to a new directory."""
    source_path = Path(source_path)
    target_path = Path(target_path)
    copy(source_path, target_path)
    