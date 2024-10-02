"""Constants for the project."""

from logging import getLogger

EPSILON = 1e-8
LOGGER = getLogger(__name__)
DEFAULT_OPTIMIZER_CONFIG = {
    "name": "adamw",  # optimizer name
    "lr": 2e-5,  # learning rate
    "mode": "min",  # mode for ReduceLROnPlateau
    "factor": 0.5,  # factor for ReduceLROnPlateau
    "patience": 5,  # patience for ReduceLROnPlateau
}
DEFAULT_METRICS = [
    "mse",
    "mae",
    "ncc",
    "psnr",
    "ssim",
    "ms_ssim",
    "niqe",
    "brisque",
]
