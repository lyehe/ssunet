"""Single volume dataset."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as tnf
import torchvision.transforms.v2.functional as tf
from numpy.random import rand, randint
from torch.utils.data import Dataset

from ..constants import EPSILON, LOGGER
from ..exceptions import (
    ShapeMismatchError,
    UnsupportedDataTypeError,
    UnsupportedInputModeError,
)


def _lucky(factor: float = 0.5) -> bool:
    """Check if you are lucky."""
    return rand() < factor


@dataclass
class SSUnetData:
    """Data class for the input data of a single volume dataset."""

    data: np.ndarray | torch.Tensor
    reference: np.ndarray | torch.Tensor | None = None

    def __post_init__(self):
        """Post initialization function."""
        # Check if the data and reference shapes match
        if self.reference is not None:
            data_shape = self.data.shape if isinstance(self.data, np.ndarray) else self.data.size()
            reference_shape = (
                self.reference.shape
                if isinstance(self.reference, np.ndarray)
                else self.reference.size()
            )
            if data_shape != reference_shape:
                raise ShapeMismatchError()

    @staticmethod
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

    def _apply_binning(self, data: np.ndarray | torch.Tensor, bin: int, mode: str) -> torch.Tensor:
        """Apply binning to the input data."""
        if isinstance(data, np.ndarray):
            data = self._to_tensor(data)

        if isinstance(data, torch.Tensor):
            if mode == "sum":
                weight = torch.ones(1, 1, bin, bin, device=data.device)
                return tnf.conv2d(data, weight, stride=bin, groups=data.size(1))
            elif mode == "max":
                return tnf.max_pool2d(data, kernel_size=bin, stride=bin)
            else:
                raise UnsupportedInputModeError()

    def binxy(self, bin: int = 2, mode: str = "sum") -> None:
        """Apply binning to the input data."""
        self.data = self._apply_binning(self.data, bin, mode=mode)
        if self.reference is not None:
            self.reference = self._apply_binning(self.reference, bin, mode=mode)


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


class SingleVolumeDataset(Dataset, ABC):
    """Single volume dataset."""

    def __init__(
        self,
        input: SSUnetData,
        config: DataConfig,
        **kwargs,
    ) -> None:
        """Initialize the single volume dataset."""
        super().__init__()
        self.input = input
        self.config = config
        self.crop_idx: tuple[int, int, int, int]
        self.kwargs = kwargs
        self.__post_init__()

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return self.length

    def __post_init__(self):
        """Post initialization function."""
        pass

    @property
    @abstractmethod
    def data_size(self) -> int:
        """Function to define the number of samples in a volume."""

    @property
    def data(self) -> torch.Tensor:
        """Get the data tensor."""
        if isinstance(self.input.data, np.ndarray):
            self.input.data = self.input._to_tensor(self.input.data)
        return self.input.data

    @property
    def reference(self) -> torch.Tensor | None:
        """Get the reference tensor."""
        if isinstance(self.input.reference, np.ndarray):
            self.input.reference = self.input._to_tensor(self.input.reference)
        return self.input.reference

    @property
    def x_size(self) -> int:
        """Get the x size."""
        return self.config.xy_size

    @property
    def y_size(self) -> int:
        """Get the y size."""
        return self.config.xy_size

    @property
    def z_size(self) -> int:
        """Get the z size."""
        return self.config.z_size

    @property
    def length(self) -> int:
        """Get the length of the dataset."""
        return (
            self.data_size if self.config.virtual_size == 0 else self.config.virtual_size
        ) // self.config.skip_frames

    @staticmethod
    def normalize_by_mean(input: torch.Tensor) -> torch.Tensor:
        """Normalize the input data by the mean."""
        return input / (input.mean() + EPSILON)

    def _new_crop_params(self) -> tuple[int, int, int, int]:
        """Compute the coordinates for the new crop window."""
        if self.config.random_crop:  # Random crop
            xi = randint(self.data.shape[-2] - self.x_size)
            yi = randint(self.data.shape[-1] - self.y_size)
            xe = xi + self.x_size
            ye = yi + self.y_size
        else:  # Center crop
            xi = (self.data.shape[-2] - self.x_size) // 2
            yi = (self.data.shape[-1] - self.y_size) // 2
            xe = xi + self.x_size
            ye = yi + self.y_size
        return xi, yi, xe, ye

    def _crop_list(self, input: list[torch.Tensor]) -> list[torch.Tensor]:
        """Crop the input data to the window size."""
        self.crop_idx = self._new_crop_params()
        return [
            data[
                ...,
                self.crop_idx[0] : self.crop_idx[2],
                self.crop_idx[1] : self.crop_idx[3],
            ]
            for data in input
        ]

    def _rotate_list(self, input: list[torch.Tensor]) -> list[torch.Tensor]:
        """Rotate the input data if the rotation flag is set to True."""
        if self.config.rotation > 0:
            angle = rand() * self.config.rotation
            return [tf.rotate(data, angle) for data in input]
        else:
            return input

    def _augment_list(self, input: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply the augmentation to the output data if the augment flag is set to True."""
        if self.config.augments:
            if _lucky():
                input = [torch.transpose(data, -1, -2) for data in input]
            if _lucky():
                input = [torch.flip(data, [-1]) for data in input]
            if _lucky():
                input = [torch.flip(data, [-2]) for data in input]
        return input

    @staticmethod
    def _add_channel_dim(input: list[torch.Tensor]) -> list[torch.Tensor]:
        """Add the channel dimension to the input data."""
        if len(input[0].shape) == 3:
            return [data.unsqueeze(0) for data in input]
        else:
            return [data.swapaxes(0, 1) for data in input]

    def _index(self, index: int) -> int:
        """Compute the true index for the input data."""
        # Random index if the virtual size is not set
        index = index if self.config.virtual_size == 0 else randint(self.data_size)
        if self.config.skip_frames > 1:
            index = index // self.config.skip_frames * self.config.skip_frames
        return index
