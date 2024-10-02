"""N2N dataset."""

import torch

from ..constants import LOGGER
from .singlevolume import SingleVolumeDataset


class MissingReferenceError(ValueError):
    """Exception raised when reference data is required."""

    def __init__(self):
        super().__init__("Reference data is required")
        LOGGER.error("MissingReferenceError: Reference data is required")


class N2NDatasetSkipFrame(SingleVolumeDataset):
    """N2N using even and odd frames as input/target."""

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """Get a N2N sample."""
        start_index = self._index(index)
        end_index = start_index + self.z_size * 2
        output = self.data[start_index:end_index]
        out_even = output[::2].float()
        out_odd = output[1::2].float()
        if self.reference is not None:
            ground_truth = self.reference[start_index:end_index:2].float()
            output = [out_odd, out_even, ground_truth]
        else:
            output = [out_odd, out_even]
        output = self._crop_list(self._rotate_list(output))
        return self._add_channel_dim(self._augment_list(output))

    @property
    def data_size(self) -> int:
        """Get the length of the dataset."""
        return self.data.shape[0] - self.z_size * 2


class N2NDatasetDualVolume(SingleVolumeDataset):
    """N2N using 2 different volumes. The target volume is the reference."""

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """Get a N2N sample."""
        if self.reference is None:
            raise MissingReferenceError()

        start_index = self._index(index)
        end_index = start_index + self.z_size
        output = [
            self.data[start_index:end_index],
            self.reference[start_index:end_index],
        ]
        output = self._crop_list(self._rotate_list(output))
        return self._add_channel_dim(self._augment_list(output))

    @property
    def data_size(self) -> int:
        """Get the length of the dataset."""
        return self.data.shape[0] - self.z_size + 1
