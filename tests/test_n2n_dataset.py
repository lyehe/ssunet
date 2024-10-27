"""Test the N2N dataset classes."""

import pytest
import torch

from src.ssunet.configs import DataConfig, SSUnetData
from src.ssunet.datasets import N2NSkipFrameDataset, PairedDataset
from src.ssunet.exceptions import MissingReferenceError


def test_n2n_dataset_skip_frame():
    """Test N2NSkipFrameDataset."""
    data_config = DataConfig(xy_size=64, z_size=32)

    data = torch.randn(100, 128, 128, device="cpu")
    input_data = SSUnetData(data)

    dataset = N2NSkipFrameDataset(input_data, data_config)
    item = dataset[0]

    assert len(item) == 2  # [odd_frames, even_frames]
    assert item[0].shape == (1, 32, 64, 64)
    assert item[1].shape == (1, 32, 64, 64)
    assert item[0].device.type == "cpu"
    assert item[1].device.type == "cpu"


def test_n2n_dataset_skip_frame_with_reference():
    """Test N2NSkipFrameDataset with reference data."""
    data_config = DataConfig(xy_size=64, z_size=32)

    data = torch.randn(100, 128, 128)
    reference = torch.randn(100, 128, 128)
    input_data = SSUnetData(data, reference)

    dataset = N2NSkipFrameDataset(input_data, data_config)
    item = dataset[0]

    assert len(item) == 3  # [odd_frames, even_frames, ground_truth]
    assert item[0].shape == (1, 32, 64, 64)
    assert item[1].shape == (1, 32, 64, 64)
    assert item[2].shape == (1, 32, 64, 64)


def test_n2n_dataset_dual_volume():
    """Test PairedDataset."""
    data_config = DataConfig(xy_size=64, z_size=32)

    data = torch.randn(100, 128, 128)
    reference = torch.randn(100, 128, 128)
    input_data = SSUnetData(data, reference)

    dataset = PairedDataset(input_data, data_config)
    item = dataset[0]

    assert len(item) == 2  # [input, target]
    assert item[0].shape == (1, 32, 64, 64)
    assert item[1].shape == (1, 32, 64, 64)


def test_n2n_dataset_dual_volume_missing_reference():
    """Test PairedDataset with missing reference data."""
    data_config = DataConfig(xy_size=64, z_size=32)

    data = torch.randn(100, 128, 128)
    input_data = SSUnetData(data)
    dataset = PairedDataset(input_data, data_config)

    with pytest.raises(MissingReferenceError):
        _ = dataset.reference


def test_n2n_dataset_skip_frame_data_size():
    """Test N2NDatasetSkipFrame data_size property."""
    data_config = DataConfig(xy_size=64, z_size=32)

    data = torch.randn(100, 128, 128)
    input_data = SSUnetData(data)

    dataset = N2NSkipFrameDataset(input_data, data_config)
    assert dataset.data_size == 36  # 100 - 32 * 2 + 1


def test_n2n_dataset_dual_volume_data_size():
    """Test PairedDataset data_size property."""
    data_config = DataConfig(xy_size=64, z_size=32)

    data = torch.randn(100, 128, 128)
    reference = torch.randn(100, 128, 128)
    input_data = SSUnetData(data, reference)

    dataset = PairedDataset(input_data, data_config)
    assert dataset.data_size == 69  # 100 - 32 + 1
