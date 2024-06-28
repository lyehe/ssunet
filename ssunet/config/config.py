from ssunet.dataloader import SingleVolumeConfig, SplitParams
from ssunet.models import ModelConfig
from ssunet.train import LoaderConfig, TrainConfig

import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PathConfig:
    """Configuration for paths"""

    dir_path: Path | None = None
    num_of_files: int | None = None
    data_dir: Path | None = None
    data_path: str | None = None
    data_file: str | None = None
    ground_truth_path: str | None = None
    ground_truth_file: str | None = None
    model_path: Path | None = None
    data_type: str | None = None


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
) -> tuple[
    PathConfig, SingleVolumeConfig, SplitParams, ModelConfig, LoaderConfig, TrainConfig
]:
    """Convert the configuration dictionary to dataclasses"""
    config = load_yaml(config_path)
    return (
        PathConfig(**config["PATH"]),
        SingleVolumeConfig(**config["DATA"]),
        SplitParams(**config["SPLIT"]),
        ModelConfig(**config["MODEL"]),
        LoaderConfig(**config["LOADER"]),
        TrainConfig(**config["TRAIN"]),
    )
