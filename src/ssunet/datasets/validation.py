"""Validation dataset."""

import torch

from ..constants import EPSILON
from ..utils import _normalize_by_mean
from .singlevolume import SingleVolumeDataset


class ValidationDataset(SingleVolumeDataset):
    """A dataset class for validation data."""

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """Get a validation sample."""
        index = self._index(index)
        input = self.data[index : index + self.z_size]

        # If secondary data is provided, use it as reference
        if self.secondary_data is not None:
            reference = self.secondary_data[index : index + self.z_size]
            target = _normalize_by_mean(input) if self.config.normalize_target else input
            output = self._crop_list_items([input / (input.mean() + EPSILON), input, reference])
        # If no secondary data is provided, use the normalized input as target
        else:
            [input] = self._crop_list_items([input])
            target = _normalize_by_mean(input) if self.config.normalize_target else input
            output = [target, input]

        return self._add_channel_dim(self._augment_list(output))

    @property
    def data_size(self) -> int:
        """Get the data size."""
        return self.data.shape[0] - self.z_size + 1
