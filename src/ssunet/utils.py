"""Utility functions."""

from pathlib import Path

import numpy as np
import torch
import yaml

from .constants import LOGGER
from .exceptions import ConfigFileNotFoundError, UnsupportedDataTypeError


def _to_tensor(input: np.ndarray) -> torch.Tensor:
    """Convert the input data to a tensor."""
    try:
        return torch.from_numpy(input)
    except TypeError:
        LOGGER.warning("Data type not supported")
        try:
            LOGGER.info("Trying to convert to int64")
            return torch.from_numpy(input.astype(np.int64))
        except TypeError as err:
            LOGGER.error("Data type not supported")
            raise UnsupportedDataTypeError() from err


def _lucky(factor: float = 0.5) -> bool:
    """Check if you are lucky."""
    return np.random.rand() < factor


def load_yaml(config_path: Path | str) -> dict:
    """Load the yaml configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigFileNotFoundError(config_path)
    return yaml.safe_load(config_path.read_text())
