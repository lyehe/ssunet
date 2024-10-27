"""Centralized error and exception definitions for the SSUnet project."""

import traceback
from enum import Enum
from pathlib import Path

from .constants import LOGGER


class SSUnetError(Exception):
    """Base class for all SSUnet errors.

    :param message: The error message
    :param include_traceback: Whether to include the traceback in the log message
    """

    def __init__(self, message: str, include_traceback: bool = False) -> None:
        super().__init__(message)
        log_message = f"{self.__class__.__name__}: {message}"
        if include_traceback:
            log_message += f"\n{traceback.format_exc()}"
        LOGGER.error(log_message)


class ConfigError(SSUnetError):
    """Base class for configuration errors."""


class DataError(SSUnetError):
    """Base class for data-related errors."""


class ModelError(SSUnetError):
    """Base class for model-related errors."""


class InferenceError(SSUnetError):
    """Base class for inference-related errors."""


class ConfigFileNotFoundError(ConfigError):
    """Error raised when the config file is not found."""

    def __init__(self, config_path: Path):
        super().__init__(f"Config file not found at {config_path}")


class ShapeMismatchError(DataError):
    """Error raised when data and reference shapes do not match."""

    def __init__(self):
        super().__init__("Data and reference shapes do not match")


class ImageShapeMismatchError(DataError):
    """Exception raised when the shapes of the image and target do not match."""

    def __init__(self):
        super().__init__("Image and target shapes must match.")


class UnsupportedDataTypeError(DataError):
    """Error raised when data type is not supported."""

    def __init__(self):
        super().__init__("Data type not supported")


class UnsupportedInputModeError(DataError):
    """Error raised when input mode is not supported."""

    def __init__(self):
        super().__init__("Input mode not supported")


class InvalidDataDimensionError(DataError):
    """Error raised when the input data has invalid dimensions."""

    def __init__(self):
        super().__init__("Data must be 3D or 4D")


class InvalidImageDimensionError(DataError):
    """Exception raised when the image is neither grayscale nor RGB."""

    def __init__(self):
        super().__init__("Image must be grayscale or RGB.")


class InvalidStackDimensionError(DataError):
    """Exception raised when the image stack is not a 3D tensor."""

    def __init__(self):
        super().__init__("Image must be a 3D tensor.")


class InvalidPValueError(DataError):
    """Error raised when p value is invalid."""

    def __init__(self, message: str):
        super().__init__(message)


class MissingPListError(DataError):
    """Error raised when p_list is missing for list method."""

    def __init__(self):
        super().__init__("p_list must be provided when method is list")


class MissingReferenceError(DataError):
    """Error raised when reference data is required."""

    def __init__(self):
        super().__init__("Reference data is required")


# Model Errors
class InvalidUpModeError(ModelError):
    """Error raised when the up mode is invalid."""

    def __init__(self, mode: str):
        super().__init__(f'Up mode "{mode}" is incompatible with merge_mode "add"')


# Inference Errors
class PatchSizeTooLargeError(InferenceError):
    """Error raised when the patch size is too large for available VRAM."""

    def __init__(self):
        super().__init__("Patch size too large for available VRAM")


class InvalidPatchValuesError(InferenceError):
    """Error raised when patch values are too small."""

    def __init__(self):
        super().__init__("Patch values are too small")


# Module-specific Errors
class InvalidInputShapeError(ModelError):
    """Error raised when the input shape is invalid."""

    def __init__(self, dim: int, shape: tuple):
        super().__init__(f"Input must be {dim}D, but got {len(shape)}D")


class PixelShuffleError(ModelError):
    """Base class for PixelShuffle errors."""


class InputDimensionError(PixelShuffleError):
    """Error raised when input tensor has incorrect dimensions."""

    def __init__(self, expected_dim: int, actual_dim: int):
        super().__init__(f"Input tensor must be {expected_dim}D, but got {actual_dim}D")


class ChannelDivisibilityError(PixelShuffleError):
    """Error raised when input channels are not divisible by scale."""

    def __init__(self, channels: int, dims: int):
        super().__init__(f"Input channels must be divisible by scale^{dims} but got {channels}")


class SizeDivisibilityError(PixelShuffleError):
    """Error raised when input size is not divisible by scale."""

    def __init__(self, sizes: tuple):
        super().__init__(f"Size must be divisible by scale, but got {', '.join(map(str, sizes))}")


# Add the following exceptions from config.py:


class DirectoryNotFoundError(SSUnetError):
    """Error raised when a directory is not found."""

    def __init__(self, directory: Path):
        super().__init__(f"Directory {directory} does not exist")


class FileIndexOutOfRangeError(SSUnetError):
    """Error raised when a file index is out of range."""

    def __init__(self, file_type: Enum, index: int):
        super().__init__(f"{file_type.name} file index {index} out of range")


class FileNotFoundError(SSUnetError):
    """Error raised when a file is not found."""

    def __init__(self, file_type: Enum, file_path: Path):
        super().__init__(f"{file_type.name} file {file_path} does not exist")


class InvalidSliceRangeError(SSUnetError):
    """Error raised when an invalid slice range is provided."""

    def __init__(self, attr: str, begin: int, end: int):
        super().__init__(f"Invalid slice range for {attr}: {begin}:{end}")


class UnknownFileTypeError(SSUnetError):
    """Error raised when an unknown file type is encountered."""

    def __init__(self, file_path: Path):
        super().__init__(f"Unknown file type for path {file_path}")


class InvalidHDF5DatasetError(SSUnetError):
    """Error raised when an HDF5 file does not contain the expected dataset."""

    def __init__(self):
        super().__init__("HDF5 file does not contain expected dataset")


class NoDataFileAvailableError(SSUnetError):
    """Error raised when no data file is available."""

    def __init__(self):
        super().__init__("No data file available")


class SSUnetDataError(DataError):
    """Base class for SSUnetData errors."""


class SingleVolumeDatasetError(DataError):
    """Base class for SingleVolumeDataset errors."""
