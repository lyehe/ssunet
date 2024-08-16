from ssunet.dataloader import SingleVolumeConfig, SplitParams
from ssunet.models import ModelConfig
from ssunet.train import LoaderConfig, TrainConfig

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import TypeVar


@dataclass
class PathConfig:
    """Configuration for paths"""

    dir_path: str | None = None
    num_of_files: int | None = None
    data_dir: Path | None = None
    data_path: str | None = None
    data_file: str | None = None
    ground_truth_path: str | None = None
    ground_truth_file: str | None = None
    model_path: str | None = None
    data_type: str | None = None


@dataclass
class MasterConifg:
    """Cinfiguration class containing all configurations"""

    path_config: PathConfig = PathConfig()
    data_config: SingleVolumeConfig = SingleVolumeConfig()
    split_params: SplitParams = SplitParams()
    model_config: ModelConfig = ModelConfig()
    loader_config: LoaderConfig = LoaderConfig()
    train_config: TrainConfig = TrainConfig()

    def _as_dict(self):
        return {
            "path_config": self.path_config,
            "data_config": self.data_config,
            "split_params": self.split_params,
            "model_config": self.model_config,
            "loader_config": self.loader_config,
            "train_config": self.train_config,
        }


def load_yaml(config_path: Path | str = Path("./config.yml")) -> dict:
    """Load yaml configuration file"""
    if isinstance(config_path, str):
        config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_config(
    config_path: Path | str = Path("./config.yml"),
) -> MasterConifg:
    """Convert the configuration dictionary to dataclasses"""
    config = load_yaml(config_path)
    return MasterConifg(
        path_config=PathConfig(**config["PATH"]),
        data_config=SingleVolumeConfig(**config["DATA"]),
        split_params=SplitParams(**config["SPLIT"]),
        model_config=ModelConfig(**config["MODEL"]),
        loader_config=LoaderConfig(**config["LOADER"]),
        train_config=TrainConfig(**config["TRAIN"]),
    )


def dump_config(
    config: MasterConifg, config_path: Path = Path("./dumped_config.yml")
) -> None:
    """Dump the MasterConfig object to a yaml file"""
    config_dict = config._as_dict()
    with open(config_path, "w") as file:
        yaml.dump(config_dict, file)
        

