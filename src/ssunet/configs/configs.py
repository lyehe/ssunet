"""Configuration for the project."""

from dataclasses import dataclass
from pathlib import Path
from shutil import copy

import yaml

from ..utils import load_yaml
from .data_config import DataConfig
from .file_config import PathConfig, SplitParams
from .model_config import ModelConfig
from .train_config import LoaderConfig, TrainConfig

example_config_path = Path("./config.yml")


@dataclass
class MasterConfig:
    """Configuration class containing all configurations."""

    data_config: DataConfig
    path_config: PathConfig
    split_params: SplitParams
    model_config: ModelConfig
    loader_config: LoaderConfig
    train_config: TrainConfig

    def _as_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "path_config": self.path_config,
            "data_config": self.data_config,
            "split_params": self.split_params,
            "model_config": self.model_config,
            "loader_config": self.loader_config,
            "train_config": self.train_config,
        }

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


def load_config(config_path: str | Path = example_config_path) -> MasterConfig:
    """Convert the configuration dictionary to dataclasses."""
    config = load_yaml(config_path)
    master_config = MasterConfig(
        path_config=PathConfig(**config["PATH"]),
        data_config=DataConfig(**config["DATA"]),
        split_params=SplitParams(**config["SPLIT"]),
        model_config=ModelConfig(**config["MODEL"]),
        loader_config=LoaderConfig(**config["LOADER"]),
        train_config=TrainConfig(**config["TRAIN"]),
    )
    master_config.train_config.set_new_root(new_root=master_config.name)
    save_config(config, master_config.train_config.default_root_dir)
    return master_config


def copy_config(source_path: Path | str, target_path: Path | str) -> None:
    """Copy the configuration to a new directory."""
    source_path = Path(source_path)
    target_path = Path(target_path)
    copy(source_path, target_path)


def save_config(config: dict, path: Path | str) -> None:
    """Save the configuration to a yaml file."""
    path = Path(path) / "config.yml"
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(config, path.open("w"))
