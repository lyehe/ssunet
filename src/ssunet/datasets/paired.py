"""Paired dataset."""

import numpy as np
import torch

from ..exceptions import MissingReferenceError
from ..utils import _to_tensor
from .singlevolume import SingleVolumeDataset


class PairedDataset(SingleVolumeDataset):
    """Paired dataset using 2 different volumes. The target volume is the reference."""

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """Get a N2N sample."""
        start_index = self._index(index)
        end_index = start_index + self.z_size
        output = [
            self.data[start_index:end_index],
            self.reference[start_index:end_index],
        ]
        output = self._crop_list_items(self._rotate_list(output))
        return self._add_channel_dim(self._augment_list(output))

    @property
    def data_size(self) -> int:
        """Get the length of the dataset."""
        return self.data.shape[0] - self.z_size + 1

    @property
    def reference(self) -> torch.Tensor:
        """Get the reference tensor."""
        if self.input.secondary_data is None:
            raise MissingReferenceError()
        if isinstance(self.input.secondary_data, np.ndarray):
            self.input.secondary_data = _to_tensor(self.input.secondary_data)
        return self.input.secondary_data
