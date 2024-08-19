from ssunet.dataloader import DataConfig, SSUnetData, SplitParams
from ssunet.models import ModelConfig
from ssunet.train import LoaderConfig, TrainConfig

import yaml
import h5py
import numpy as np
from typing import Callable
from pathlib import Path
from itertools import islice
from dataclasses import dataclass
from tifffile import imread, TiffFile


@dataclass
class PathConfig:
    """Configuration for paths"""

    model_path: str | Path
    data_dir: str | Path
    data_file: int | str | Path

    reference_dir: str | Path | None = None
    reference_file: int | str | Path | None = None
    ground_truth_dir: str | Path | None = None
    ground_truth_file: int | str | Path | None = None

    begin_slice: int = 0
    end_slice: int = -1

    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self.data_dir = Path(self.data_dir)
        self.reference_dir = Path(self.reference_dir) if self.reference_dir else None
        self.ground_truth_dir = (
            Path(self.ground_truth_dir) if self.ground_truth_dir else None
        )

        # Verify or create the model path
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Verify the data directory exists
        if not self.data_dir.exists():
            raise NotADirectoryError(f"Data directory {self.data_dir} does not exist")
        self.data_file = self._resolve_file(self.data_file, self.data_dir, "Data")

        # Handle reference_file if reference_dir is provided
        if self.reference_dir and self.reference_file:
            if not self.reference_dir.exists():
                raise NotADirectoryError(
                    f"Reference directory {self.reference_dir} does not exist"
                )
            self.reference_file = self._resolve_file(
                self.reference_file, self.reference_dir, "Reference"
            )

        # Handle ground_truth_file if ground_truth_dir is provided
        if self.ground_truth_dir and self.ground_truth_file:
            if not self.ground_truth_dir.exists():
                raise NotADirectoryError(
                    f"Ground truth directory {self.ground_truth_dir} does not exist"
                )
            self.ground_truth_file = self._resolve_file(
                self.ground_truth_file, self.ground_truth_dir, "Ground_Truth"
            )

    def _resolve_file(
        self, file_input: int | str | Path, directory: Path, file_type: str
    ):
        if isinstance(file_input, int):
            try:
                return next(islice(directory.iterdir(), file_input, None))
            except StopIteration:
                raise IndexError(f"{file_type} file index {file_input} out of range")
        elif isinstance(file_input, (str, Path)):
            file_path = Path(file_input)
            if not file_path.is_absolute():
                file_path = directory / file_path
            if not file_path.exists():
                raise FileNotFoundError(f"{file_type} file {file_path} does not exist")
            return file_path

    @property
    def reference_is_avaiable(self) -> bool:
        return self.reference_dir is not None and self.reference_file is not None

    @property
    def ground_truth_is_available(self) -> bool:
        return self.ground_truth_dir is not None and self.ground_truth_file is not None

    def _load(self, path: Path, method: Callable | None) -> np.ndarray:
        if path.suffix in [".tif", ".tiff"]:
            if self.end_slice == -1:
                return imread(path)
            else:
                with TiffFile(str(path)) as tif:
                    return tif.asarray(key=slice(self.begin_slice, self.end_slice))

        elif path.suffix in [".h5", ".hdf5"]:
            with h5py.File(str(path), "r") as f:
                keys = list(f.keys())
                dataset = f.get(keys[0])
                assert isinstance(
                    dataset, h5py.Dataset
                ), "HDF5 file does not contain expected dataset"
                return np.array(dataset[self.begin_slice : self.end_slice])

        elif method is not None:
            return method(path)
        else:
            raise ValueError(f"Unknown file type for path {path}")

    def load_data(self, method: Callable | None = None) -> np.ndarray:
        if isinstance(self.data_file, Path):
            return self._load(self.data_file, method)
        else:
            raise ValueError("No data file available")

    def load_reference(self, method: Callable | None = None) -> np.ndarray | None:
        if self.reference_is_avaiable and isinstance(self.reference_file, Path):
            return self._load(self.reference_file, method)
        else:
            return None

    def load_ground_truth(self, method: Callable | None = None) -> np.ndarray | None:
        if self.ground_truth_is_available and isinstance(self.ground_truth_file, Path):
            return self._load(self.ground_truth_file, method)
        else:
            return None

    def load_ssunet_data(self, method: Callable | None = None) -> SSUnetData:
        data = self.load_data(method)
        reference = self.load_reference(method)
        return SSUnetData(data=data, reference=reference)


@dataclass
class MasterConfig:
    """Cinfiguration class containing all configurations"""

    data_config: DataConfig
    path_config: PathConfig
    split_params: SplitParams
    model_config: ModelConfig
    loader_config: LoaderConfig
    train_config: TrainConfig

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
) -> MasterConfig:
    """Convert the configuration dictionary to dataclasses"""
    config = load_yaml(config_path)
    return MasterConfig(
        path_config=PathConfig(**config["PATH"]),
        data_config=DataConfig(**config["DATA"]),
        split_params=SplitParams(**config["SPLIT"]),
        model_config=ModelConfig(**config["MODEL"]),
        loader_config=LoaderConfig(**config["LOADER"]),
        train_config=TrainConfig(**config["TRAIN"]),
    )


def dump_config(
    config: MasterConfig, config_path: Path = Path("./dumped_config.yml")
) -> None:
    """Dump the MasterConfig object to a yaml file"""
    config_dict = config._as_dict()
    with open(config_path, "w") as file:
        yaml.dump(config_dict, file)
