from dataclasses import dataclass
from abc import abstractmethod, ABC
import logging

import numpy as np
from numpy.random import randint, rand
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as tf

EPSILON = 1e-8

logger = logging.getLogger(__name__)


def lucky(factor: float = 0.5) -> bool:
    """Check if you are lucky."""
    return rand() < factor


@dataclass
class SingleVolumeConfig:
    """Data class for the input data of a single volume dataset."""

    data: np.ndarray | torch.Tensor
    xy_size: int
    z_size: int
    virtual_size: int = 0
    augments: bool = False
    rotation: float = 0
    random_crop: bool = True
    skip_frames: int = 1
    reference: np.ndarray | torch.Tensor | None = None
    normalize_target: bool = True
    note: str = ""

    def __post_init__(self):
        # Check if the data and reference shapes match
        if self.reference is not None:
            data_shape = (
                self.data.shape
                if isinstance(self.data, np.ndarray)
                else self.data.size()
            )
            reference_shape = (
                self.reference.shape
                if isinstance(self.reference, np.ndarray)
                else self.reference.size()
            )
            if data_shape != reference_shape:
                raise ValueError("Data and reference shapes do not match")

    def to_tensor(self, input: np.ndarray) -> torch.Tensor:
        try:
            return torch.from_numpy(input)
        except TypeError:
            logger.warning("Data type not supported")
            try:
                logger.info("Trying to convert to int64")
                return torch.from_numpy(input.astype(np.int64))
            except TypeError:
                logger.error("Data type not supported")
                raise TypeError("Data type not supported")

    @property
    def name(self) -> str:
        return f"{self.note}_{self.stack_size}x{self.z_size}x{self.xy_size}x{self.xy_size}_skip={self.skip_frames}"

    @property
    def stack_size(self) -> int:
        return self.virtual_size if self.virtual_size > 0 else len(self.data)

    def __call__(self):
        return self.data


class SingleVolumeDataset(Dataset, ABC):
    def __init__(
        self,
        input: SingleVolumeConfig,
        **kwargs,
    ):
        super().__init__()
        self.input = input
        self.crop_idx: tuple[int, int, int, int]
        self.kwargs = kwargs
        self.__post_init__()

    def __len__(self) -> int:
        return self.length

    def __post_init__(self):
        """Post initialization function."""
        pass

    @property
    @abstractmethod
    def data_size(self) -> int: ...

    """Function to define the number of samples in a volume."""

    @property
    def data(self) -> torch.Tensor:
        if isinstance(self.input.data, np.ndarray):
            self.input.data = self.input.to_tensor(self.input.data)
        return self.input.data

    @property
    def reference(self) -> torch.Tensor | None:
        if isinstance(self.input.reference, np.ndarray):
            self.input.reference = self.input.to_tensor(self.input.reference)
        return self.input.reference

    @property
    def x_size(self) -> int:
        return self.input.xy_size

    @property
    def y_size(self) -> int:
        return self.input.xy_size

    @property
    def z_size(self) -> int:
        return self.input.z_size

    @property
    def length(self) -> int:
        return (
            self.data_size if self.input.virtual_size == 0 else self.input.virtual_size
        ) // self.input.skip_frames

    @staticmethod
    def normalize_by_mean(input: torch.Tensor) -> torch.Tensor:
        return input / (input.mean() + EPSILON)

    def _new_crop_params(self) -> tuple[int, int, int, int]:
        """Compute the coordinates for the new crop window."""
        if self.input.random_crop:  # Random crop
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
        if self.input.rotation > 0:
            angle = rand() * self.input.rotation
            return [tf.rotate(data, angle) for data in input]
        else:
            return input

    def _augment_list(self, input: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply the augmentation to the output data if the augment flag is set to True."""
        if self.input.augments:
            if lucky():
                input = [torch.transpose(data, -1, -2) for data in input]
            if lucky():
                input = [torch.flip(data, [-1]) for data in input]
            if lucky():
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
        index = index if self.input.virtual_size == 0 else randint(self.data_size)
        if self.input.skip_frames > 1:
            index = index // self.input.skip_frames * self.input.skip_frames
        return index


class ValidationDataset(SingleVolumeDataset):
    """A dataset class for validation data."""

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        index = self._index(index)
        input = self.data[index : index + self.z_size]
        # Combine the input and ground truth data for cropping
        if self.reference is not None:
            reference = self.reference[index : index + self.z_size]
            target = (
                self.normalize_by_mean(input) if self.input.normalize_target else input
            )
            output = self._crop_list(
                [input / (input.mean() + EPSILON), input, reference]
            )
        else:
            [input] = self._crop_list([input])
            target = (
                self.normalize_by_mean(input) if self.input.normalize_target else input
            )
            output = [target, input]

        return self._add_channel_dim(self._augment_list(output))