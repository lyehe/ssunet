"""Configuration for the file import."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import islice
from pathlib import Path

import h5py
import numpy as np
from numpy.random import seed
from tifffile import TiffFile, imread

from ..constants import EPSILON, LOGGER
from ..exceptions import (
    DirectoryNotFoundError,
    FileIndexOutOfRangeError,
    FileNotFoundError,
    InvalidHDF5DatasetError,
    InvalidSliceRangeError,
    NoDataFileAvailableError,
    UnknownFileTypeError,
)
from .data_config import SSUnetData

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
    data_file: PathLike
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
        if self.data_dir and not self.data_dir.exists():
            raise DirectoryNotFoundError(self.data_dir)
        self.data_file = self._resolve_file(self.data_file, self.data_dir, FileType.DATA)

        # Handle reference_file if reference_dir is provided
        if self.reference_dir and self.reference_file:
            if not self.reference_dir.exists():
                raise DirectoryNotFoundError(self.reference_dir)
            self.reference_file = self._resolve_file(
                self.reference_file, self.reference_dir, FileType.REFERENCE
            )

        # Handle ground_truth_file if ground_truth_dir is provided
        if self.ground_truth_dir and self.ground_truth_file:
            if not self.ground_truth_dir.exists():
                raise DirectoryNotFoundError(self.ground_truth_dir)
            self.ground_truth_file = self._resolve_file(
                self.ground_truth_file, self.ground_truth_dir, FileType.GROUND_TRUTH
            )

        self._validate_slices()

    def _validate_slices(self) -> None:
        """Validate slice ranges for data, reference, and ground truth."""
        for attr in ["data", "reference", "ground_truth"]:
            begin = getattr(self, f"{attr}_begin_slice")
            end = getattr(self, f"{attr}_end_slice")
            if begin < 0 or (end != -1 and end <= begin):
                raise InvalidSliceRangeError(attr, begin, end)

    def _resolve_file(self, file_input: FileInput, directory: Path, file_type: FileType) -> Path:
        """Resolve the file path."""
        if isinstance(file_input, int):
            try:
                return next(islice(directory.iterdir(), file_input, None))
            except StopIteration as err:
                raise FileIndexOutOfRangeError(file_type, file_input) from err
        elif isinstance(file_input, str | Path):
            file_path = Path(file_input)
            if not file_path.is_absolute():
                file_path = directory / file_path
            if not file_path.exists():
                raise FileNotFoundError(file_type, file_path)
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
                    raise InvalidHDF5DatasetError()
                return np.array(dataset[begin:end])

        elif method is not None:
            return method(data_path, begin=begin, end=end)
        else:
            raise UnknownFileTypeError(data_path)

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
            raise NoDataFileAvailableError()

    def load_reference(
        self,
        method: Callable | None = None,
        begin: int | None = None,
        end: int | None = None,
    ) -> np.ndarray:
        """Load the reference file. If no reference is available, return the data."""
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
    ) -> np.ndarray:
        """Load the ground truth file. If no ground truth is available, return the reference."""
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

    def load_data_only(self, method: Callable | None = None) -> SSUnetData:
        """Load the data file only."""
        data = self.load_data(method)
        return SSUnetData(primary_data=data)

    def load_reference_only(self, method: Callable | None = None) -> SSUnetData:
        """Load the reference file only."""
        reference = self.load_reference(method)
        return SSUnetData(primary_data=reference)

    def load_data_and_ground_truth(self, method: Callable | None = None) -> SSUnetData:
        """Load SSUnetData and return a SSUnetData object."""
        data = self.load_data(method)
        reference = self.load_ground_truth(method)
        return SSUnetData(primary_data=data, secondary_data=self._normalize_ground_truth(reference))

    def load_reference_and_ground_truth(self, method: Callable | None = None) -> SSUnetData:
        """Load SSUnetData and return a SSUnetData object."""
        reference = self.load_reference(method)
        ground_truth = self.load_ground_truth(method)
        return SSUnetData(
            primary_data=reference, secondary_data=self._normalize_ground_truth(ground_truth)
        )

    def _normalize_ground_truth(self, ground_truth: np.ndarray) -> np.ndarray:
        """Normalize the ground truth to be between 0 and 1."""
        gt_max = np.max(ground_truth)
        if gt_max < 255 and gt_max > 1:
            return ground_truth.astype(np.float32) / 255
        elif gt_max > 255:
            return ground_truth.astype(np.float32) / 65535


@dataclass
class SplitParams:
    """Configuration for splitting the data into target and noise components."""

    method: str = "signal"
    min_p: float = EPSILON
    max_p: float = 1 - EPSILON
    p_list: list[float] | None = field(default_factory=list)
    normalize_target: bool = True
    seed: int | None = None

    def __post_init__(self):
        """Validate and initialize the split parameters."""
        if self.min_p > self.max_p:
            LOGGER.warning("min_p should be less than max_p, swapping values")
            self.min_p, self.max_p = self.max_p, self.min_p
        if self.seed is not None:
            seed(self.seed)
