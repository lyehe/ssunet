"""Centralized error and exception definitions for the SSUnet project."""

import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)


class SSUnetError(Exception):
    """Base class for all SSUnet errors."""

    def __init__(self, message: str):
        super().__init__(message)
        logger.error(f"{self.__class__.__name__}: {message}")


class ConfigError(SSUnetError):
    """Base class for configuration errors."""


class DataError(SSUnetError):
    """Base class for data-related errors."""


class ModelError(SSUnetError):
    """Base class for model-related errors."""


class InferenceError(SSUnetError):
    """Base class for inference-related errors."""


# Configuration Errors
class ConfigFileNotFoundError(ConfigError):
    """Error raised when the config file is not found."""

    def __init__(self, config_path: Path):
        super().__init__(f"Config file not found at {config_path}")


# Data Errors
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
