"""Utility functions."""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from .constants import EPSILON, LOGGER
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


def _load_yaml(config_path: Path | str) -> dict:
    """Load the yaml configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigFileNotFoundError(config_path)
    return yaml.safe_load(config_path.read_text())


def _normalize_by_mean(input: torch.Tensor) -> torch.Tensor:
    """Normalize the input data by the mean."""
    return input / (input.mean() + EPSILON)


def setup_logger(
    level: int | str = LOGGER.level,
    log_file: Path | None = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Set up logger with either console or file handler.

    :param level: Logging level
    :param log_file: Optional path to log file. If None, logs to console
    :param log_format: Log message format
    :param date_format: Date format in log messages
    """
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(log_format, date_format)

    if log_file is None:
        handler = logging.StreamHandler(sys.stdout)
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file)

    handler.setLevel(level)
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
