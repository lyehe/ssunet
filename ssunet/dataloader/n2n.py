import torch
import logging

from singlevolume import SingleVolumeDataset

logger = logging.getLogger(__name__)


class N2NDatasetSkipFrame(SingleVolumeDataset):
    def __getitem__(self, index: int) -> list[torch.Tensor]:
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
        return self.data.shape[0] - self.z_size * 2


class N2NDatasetDualVolume(SingleVolumeDataset):
    def __getitem__(self, index) -> list[torch.Tensor]:
        if self.reference is None:
            logger.error("Reference data is required for the dual N2N dataset")
            raise ValueError("Reference data is required for the dual N2N dataset")
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
        return self.data.shape[0] - self.z_size + 1
