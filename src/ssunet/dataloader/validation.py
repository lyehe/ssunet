"""Validation dataset."""

import torch

from ..constants import EPSILON
from .singlevolume import SingleVolumeDataset


class ValidationDataset(SingleVolumeDataset):
    """A dataset class for validation data."""

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """Get a validation sample."""
        index = self._index(index)
        input = self.data[index : index + self.z_size]
        # Combine the input and ground truth data for cropping
        if self.reference is not None:
            reference = self.reference[index : index + self.z_size]
            target = self.normalize_by_mean(input) if self.config.normalize_target else input
            output = self._crop_list([input / (input.mean() + EPSILON), input, reference])
        else:
            [input] = self._crop_list([input])
            target = self.normalize_by_mean(input) if self.config.normalize_target else input
            output = [target, input]

        return self._add_channel_dim(self._augment_list(output))
