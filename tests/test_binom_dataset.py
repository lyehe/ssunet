"""Test the BinomDataset class."""

import numpy as np
import pytest
import torch

from src.ssunet.configs import DataConfig, SplitParams, SSUnetData
from src.ssunet.datasets import BernoulliDataset, BinomDataset
from src.ssunet.exceptions import InvalidPValueError, ShapeMismatchError


def test_binom_dataset_initialization():
    """Test the initialization of BinomDataset."""
    data_config = DataConfig(xy_size=128, z_size=32)
    split_params = SplitParams(method="signal", min_p=0.1, max_p=0.9)

    # Create a dummy data tensor
    data = torch.randn(100, 128, 128)
    input_data = SSUnetData(data)

    dataset = BinomDataset(input_data, data_config, split_params=split_params)
    assert len(dataset) == 69  # 100 - 32 + 1


def test_binom_dataset_getitem():
    """Test the __getitem__ method of BinomDataset."""
    data_config = DataConfig(xy_size=64, z_size=32)
    split_params = SplitParams(method="signal", min_p=0.1, max_p=0.9)

    # Create a dummy data tensor
    data = torch.randn(100, 128, 128)
    input_data = SSUnetData(data)

    dataset = BinomDataset(input_data, data_config, split_params=split_params)
    item = dataset[0]

    assert len(item) == 2  # [target, noise]
    assert item[0].shape == (1, 32, 64, 64)
    assert item[1].shape == (1, 32, 64, 64)


def test_binom_dataset_invalid_p_value():
    """Test BinomDataset with invalid p-value."""
    data_config = DataConfig(xy_size=128, z_size=32)
    split_params = SplitParams(method="fixed", min_p=0, max_p=1)

    # Create a dummy data tensor
    data = torch.randn(100, 128, 128)
    input_data = SSUnetData(data)

    with pytest.raises(InvalidPValueError):
        BinomDataset(input_data, data_config, split_params=split_params)


def test_binom_dataset_data_size():
    """Test BinomDataset with different data sizes."""
    data_config = DataConfig(xy_size=128, z_size=32)
    split_params = SplitParams(method="signal", min_p=0.1, max_p=0.9)

    # Create dummy data tensors of different sizes
    data1 = torch.randn(100, 128, 128)
    data2 = torch.randn(50, 128, 128)

    input_data1 = SSUnetData(data1)
    input_data2 = SSUnetData(data2)

    dataset1 = BinomDataset(input_data1, data_config, split_params=split_params)
    dataset2 = BinomDataset(input_data2, data_config, split_params=split_params)

    assert dataset1.data_size == 69  # 100 - 32 + 1
    assert dataset2.data_size == 19  # 50 - 32 + 1


def test_binom_dataset_with_reference():
    """Test BinomDataset with reference data."""
    data_config = DataConfig(xy_size=64, z_size=32)
    split_params = SplitParams(method="signal", min_p=0.1, max_p=0.9)

    # Create dummy data and reference tensors
    data = torch.randn(100, 128, 128)
    reference = torch.randn(100, 128, 128)
    input_data = SSUnetData(data, reference)

    dataset = BinomDataset(input_data, data_config, split_params=split_params)
    item = dataset[0]

    assert len(item) == 3  # [target, noise, reference]
    assert item[0].shape == (1, 32, 64, 64)
    assert item[1].shape == (1, 32, 64, 64)
    assert item[2].shape == (1, 32, 64, 64)


def test_binom_dataset_shape_mismatch():
    """Test BinomDataset with mismatched data and reference shapes."""
    DataConfig(xy_size=128, z_size=32)
    SplitParams(method="signal", min_p=0.1, max_p=0.9)

    # Create dummy data and reference tensors with mismatched shapes
    data = torch.randn(100, 128, 128)
    reference = torch.randn(90, 128, 128)

    with pytest.raises(ShapeMismatchError):
        SSUnetData(data, reference)


def test_binom_dataset_with_numpy_input():
    """Test BinomDataset with numpy array input."""
    data_config = DataConfig(xy_size=64, z_size=32)
    split_params = SplitParams(method="signal", min_p=0.1, max_p=0.9)

    # Create dummy numpy data
    data = np.random.randn(100, 128, 128)
    input_data = SSUnetData(data)

    dataset = BinomDataset(input_data, data_config, split_params=split_params)
    item = dataset[0]

    assert len(item) == 2  # [target, noise]
    assert item[0].shape == (1, 32, 64, 64)
    assert item[1].shape == (1, 32, 64, 64)
    assert isinstance(item[0], torch.Tensor)
    assert isinstance(item[1], torch.Tensor)


def test_binom_dataset_db_method():
    """Test BinomDataset with 'db' method."""
    data_config = DataConfig(xy_size=64, z_size=32)
    split_params = SplitParams(method="db", min_p=0.1, max_p=0.9)

    data = torch.randn(100, 128, 128)
    input_data = SSUnetData(data)

    dataset = BinomDataset(input_data, data_config, split_params=split_params)
    item = dataset[0]

    assert len(item) == 2
    assert item[0].shape == (1, 32, 64, 64)
    assert item[1].shape == (1, 32, 64, 64)


def test_binom_dataset_fixed_method():
    """Test BinomDataset with 'fixed' method."""
    data_config = DataConfig(xy_size=64, z_size=32)
    split_params = SplitParams(method="fixed", min_p=0.5, max_p=0.5, p_list=[0.5])

    data = torch.randn(100, 128, 128)
    input_data = SSUnetData(data)

    dataset = BinomDataset(input_data, data_config, split_params=split_params)
    item = dataset[0]

    assert len(item) == 2
    assert item[0].shape == (1, 32, 64, 64)
    assert item[1].shape == (1, 32, 64, 64)


def test_binom_dataset_list_method():
    """Test BinomDataset with 'list' method."""
    data_config = DataConfig(xy_size=64, z_size=32)
    split_params = SplitParams(method="list", p_list=[0.1, 0.5, 0.9])

    data = torch.randn(100, 128, 128)
    input_data = SSUnetData(data)

    dataset = BinomDataset(input_data, data_config, split_params=split_params)
    item = dataset[0]

    assert len(item) == 2
    assert item[0].shape == (1, 32, 64, 64)
    assert item[1].shape == (1, 32, 64, 64)


def test_binom_dataset_custom_p_sampling():
    """Test BinomDataset with custom p_sampling_method."""
    data_config = DataConfig(xy_size=64, z_size=32)
    split_params = SplitParams(method="signal", min_p=0.1, max_p=0.9)

    def custom_p_sampling(input: torch.Tensor, **kwargs) -> float:
        return 0.5

    data = torch.randn(100, 128, 128)
    input_data = SSUnetData(data)

    dataset = BinomDataset(
        input_data, data_config, split_params=split_params, p_sampling_method=custom_p_sampling
    )
    item = dataset[0]

    assert len(item) == 2
    assert item[0].shape == (1, 32, 64, 64)
    assert item[1].shape == (1, 32, 64, 64)


def test_bernoulli_dataset():
    """Test BernoulliDataset."""
    data_config = DataConfig(xy_size=64, z_size=32)
    split_params = SplitParams(method="signal", min_p=0.1, max_p=0.9)

    data = torch.randn(100, 128, 128)
    input_data = SSUnetData(data)

    dataset = BernoulliDataset(input_data, data_config, split_params=split_params)
    item = dataset[0]

    assert len(item) == 2
    assert item[0].shape == (1, 32, 64, 64)
    assert item[1].shape == (1, 32, 64, 64)
    assert torch.all((item[1] == 0) | (item[1] == 1))  # Check if noise is binary
