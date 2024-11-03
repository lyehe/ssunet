"""Single volume dataset."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as tnf

from ..exceptions import (
    ShapeMismatchError,
    UnsupportedInputModeError,
)
from ..utils import _to_tensor


@dataclass
class DataConfig:
    """Data class for the input data of a single volume dataset."""

    xy_size: int = 256
    z_size: int = 32
    virtual_size: int = 0
    augments: bool = False
    rotation: float = 0
    random_crop: bool = True
    skip_frames: int = 1
    normalize_target: bool = True
    note: str = ""

    @property
    def name(self) -> str:
        """Get the name of the dataset."""
        return (
            f"{self.note}_{self.virtual_size}x{self.z_size}x{self.xy_size}x{self.xy_size}_skip"
            f"={self.skip_frames}"
        )

    @property
    def validation_config(self) -> "DataConfig":
        """Get the validation configuration."""
        return DataConfig(
            xy_size=self.xy_size,
            z_size=self.z_size,
            virtual_size=0,
            augments=False,
            rotation=0,
            random_crop=False,
            skip_frames=1,
            normalize_target=False,
            note=self.note + "_validation",
        )

    @property
    def as_dict(self) -> dict:
        """Convert the dataclass to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class SSUnetData:
    """Dataclass for the input data of a single volume dataset."""

    primary_data: np.ndarray | torch.Tensor
    secondary_data: np.ndarray | torch.Tensor | None = None

    def __post_init__(self):
        """Post initialization function."""
        # Check if the primary and secondary data shapes match
        if self.secondary_data is not None:
            data_shape = (
                self.primary_data.shape
                if isinstance(self.primary_data, np.ndarray)
                else self.primary_data.size()
            )
            secondary_data_shape = (
                self.secondary_data.shape
                if isinstance(self.secondary_data, np.ndarray)
                else self.secondary_data.size()
            )
            if data_shape != secondary_data_shape:
                raise ShapeMismatchError()

    @property
    def shape(self) -> tuple:
        """Get the shape of the data."""
        return self.primary_data.shape

    @property
    def data(self) -> np.ndarray | torch.Tensor:
        """Alias for the primary data."""
        return self.primary_data

    @property
    def reference(self) -> np.ndarray | torch.Tensor | None:
        """Alias for the secondary data."""
        return self.secondary_data

    @staticmethod
    def _apply_binning(input: np.ndarray | torch.Tensor, bin: int, mode: str) -> torch.Tensor:
        """Apply binning to the input data."""
        if isinstance(input, np.ndarray):
            input = _to_tensor(input)
        if mode == "sum":
            weight = torch.ones(1, 1, bin, bin, device=input.device)
            return tnf.conv2d(input, weight, stride=bin, groups=input.size(1))
        elif mode == "max":
            return tnf.max_pool2d(input, kernel_size=bin, stride=bin)
        else:
            raise UnsupportedInputModeError()

    def binxy(self, bin: int = 2, mode: str = "sum") -> None:
        """Apply binning to the input data."""
        self.primary_data = self._apply_binning(self.primary_data, bin, mode=mode)
        if self.secondary_data is not None:
            self.secondary_data = self._apply_binning(self.secondary_data, bin, mode=mode)
