"""Configuration for the project."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from itertools import islice
from pathlib import Path

import h5py
import numpy as np
import yaml
from tifffile import TiffFile, imread

from ssunet.dataloader import DataConfig, SplitParams, SSUnetData
from ssunet.models import ModelConfig
from ssunet.train import LoaderConfig, TrainConfig

PathLike = str | Path
FileInput = int | str | Path


class FileType(Enum):
    """Enum for file types."""

    DATA = auto()
    REFERENCE = auto()
    GROUND_TRUTH = auto()


@dataclass
class PathConfig:
    """Configuration for paths."""

    data_dir: PathLike
    data_file: FileInput
    data_begin_slice: int = 0
    data_end_slice: int = -1

    reference_dir: PathLike | None = None
    reference_file: FileInput | None = None
    reference_begin_slice: int = 0
    reference_end_slice: int = -1

    ground_truth_dir: PathLike | None = None
    ground_truth_file: FileInput | None = None
    ground_truth_begin_slice: int = 0
    ground_truth_end_slice: int = -1

    def __post_init__(self):
        """Initialize and validate paths and files."""
        self.data_dir = Path(self.data_dir)
        self.reference_dir = Path(self.reference_dir) if self.reference_dir else None
        self.ground_truth_dir = Path(self.ground_truth_dir) if self.ground_truth_dir else None

        # Verify the data directory exists
        if not self.data_dir.exists():
            raise NotADirectoryError(f"Data directory {self.data_dir} does not exist")
        self.data_file = self._resolve_file(self.data_file, self.data_dir, FileType.DATA)

        # Handle reference_file if reference_dir is provided
        if self.reference_dir and self.reference_file:
            if not self.reference_dir.exists():
                raise NotADirectoryError(f"Reference directory {self.reference_dir} does not exist")
            self.reference_file = self._resolve_file(
                self.reference_file, self.reference_dir, FileType.REFERENCE
            )

        # Handle ground_truth_file if ground_truth_dir is provided
        if self.ground_truth_dir and self.ground_truth_file:
            if not self.ground_truth_dir.exists():
                raise NotADirectoryError(
                    f"Ground truth directory {self.ground_truth_dir} does not exist"
                )
            self.ground_truth_file = self._resolve_file(
                self.ground_truth_file, self.ground_truth_dir, FileType.GROUND_TRUTH
            )

        self._validate_slices()

    def _validate_slices(self):
        """Validate slice ranges for data, reference, and ground truth."""
        for attr in ["data", "reference", "ground_truth"]:
            begin = getattr(self, f"{attr}_begin_slice")
            end = getattr(self, f"{attr}_end_slice")
            if begin < 0 or (end != -1 and end <= begin):
                raise ValueError(f"Invalid slice range for {attr}: {begin}:{end}")

    def _resolve_file(self, file_input: FileInput, directory: Path, file_type: FileType) -> Path:
        """Resolve the file path."""
        if isinstance(file_input, int):
            try:
                return next(islice(directory.iterdir(), file_input, None))
            except StopIteration as err:
                raise ValueError(f"{file_type.name} file index {file_input} out of range") from err
        elif isinstance(file_input, str | Path):
            file_path = Path(file_input)
            if not file_path.is_absolute():
                file_path = directory / file_path
            if not file_path.exists():
                raise FileNotFoundError(f"{file_type.name} file {file_path} does not exist")
            return file_path

    @property
    def reference_is_available(self) -> bool:
        """Check if reference file is available."""
        return self.reference_dir is not None and self.reference_file is not None

    @property
    def ground_truth_is_available(self) -> bool:
        """Check if ground truth file is available."""
        return self.ground_truth_dir is not None and self.ground_truth_file is not None

    def _load(
        self,
        data_path: Path,
        method: Callable | None,
        begin: int,
        end: int,
    ) -> np.ndarray:
        """Load data from file."""
        if data_path.suffix in [".tif", ".tiff"]:
            if begin == 0 and end == -1:
                return imread(data_path)
            else:
                with TiffFile(str(data_path)) as tif:
                    return tif.asarray(key=slice(begin, end))

        elif data_path.suffix in [".h5", ".hdf5"]:
            with h5py.File(str(data_path), "r") as f:
                keys = list(f.keys())
                dataset = f.get(keys[0])

                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError("HDF5 file does not contain expected dataset")
                return np.array(dataset[begin:end])

        elif method is not None:
            return method(data_path, begin=begin, end=end)
        else:
            raise ValueError(f"Unknown file type for path {data_path}")

    def load_data(
        self,
        method: Callable | None = None,
        begin: int | None = None,
        end: int | None = None,
    ) -> np.ndarray:
        """Load the data file."""
        if begin is None:
            begin = self.data_begin_slice
        if end is None:
            end = self.data_end_slice
        if isinstance(self.data_file, Path):
            return self._load(
                self.data_file,
                method,
                begin,
                end,
            ).astype(np.float32)
        else:
            raise ValueError("No data file available")

    def load_reference(
        self,
        method: Callable | None = None,
        begin: int | None = None,
        end: int | None = None,
    ) -> np.ndarray | None:
        """Load the reference file."""
        if begin is None:
            begin = self.reference_begin_slice
        if end is None:
            end = self.reference_end_slice
        if self.reference_is_available and isinstance(self.reference_file, Path):
            return self._load(
                self.reference_file,
                method,
                begin,
                end,
            ).astype(np.float32)
        else:
            return self.load_data(method)

    def load_ground_truth(
        self,
        method: Callable | None = None,
        begin: int | None = None,
        end: int | None = None,
    ) -> np.ndarray | None:
        """Load the ground truth file."""
        if begin is None:
            begin = self.ground_truth_begin_slice
        if end is None:
            end = self.ground_truth_end_slice
        if self.ground_truth_is_available and isinstance(self.ground_truth_file, Path):
            return self._load(
                self.ground_truth_file,
                method,
                begin,
                end,
            ).astype(np.float32)
        else:
            return self.load_reference(method)

    def load_ssunet_data(self, method: Callable | None = None) -> SSUnetData:
        """Load SSUnetData."""
        data = self.load_data(method)
        reference = self.load_reference(method)
        return SSUnetData(data=data, reference=reference)


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


def load_yaml(config_path: Path | str = Path("./config.yml")) -> dict:
    """Load yaml configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    return yaml.safe_load(config_path.read_text())


def load_config(
    config_path: Path | str = Path("./config.yml"),
) -> MasterConfig:
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


def save_config(config: dict, path: Path | str) -> None:
    """Save the configuration to a yaml file."""
    path = Path(path) / "config.yml"
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(config, path.open("w"))
