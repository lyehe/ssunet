"""Single volume dataset."""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torchvision.transforms.v2.functional as tf
from numpy.random import rand, randint
from torch.utils.data import Dataset

from ..configs import DataConfig, SSUnetData
from ..utils import _lucky, _to_tensor


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
        if isinstance(self.input.primary_data, np.ndarray):
            self.input.primary_data = _to_tensor(self.input.primary_data)
        return self.input.primary_data

    @property
    def secondary_data(self) -> torch.Tensor | None:
        """Get the reference tensor."""
        if isinstance(self.input.secondary_data, np.ndarray):
            self.input.secondary_data = _to_tensor(self.input.secondary_data)
        return self.input.secondary_data

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

    def _crop_list_items(self, input: list[torch.Tensor]) -> list[torch.Tensor]:
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
