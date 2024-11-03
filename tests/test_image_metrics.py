import numpy as np
import pytest
import torch

from tools.metrics import ImageMetrics


@pytest.fixture
def image_metrics():
    """Create and return an ImageMetrics instance for testing."""
    image = torch.rand(128, 128, device="cpu")
    target = torch.rand(128, 128, device="cpu")
    return ImageMetrics(image, target)


def test_image_metrics_initialization(image_metrics: ImageMetrics):
    """Test if ImageMetrics is initialized correctly."""
    assert isinstance(image_metrics, ImageMetrics)


def test_image_metrics_to_tensor(image_metrics: ImageMetrics):
    """Test the _to_tensor method for both numpy arrays and torch tensors."""
    # Test with numpy array
    np_array = np.random.rand(32, 128, 128).astype(np.float32)
    tensor = image_metrics._to_tensor(np_array)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (32, 128, 128)
    assert tensor.device.type == "cpu"

    # Test with torch tensor
    torch_tensor = torch.rand(32, 128, 128, device="cpu")
    tensor = image_metrics._to_tensor(torch_tensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (32, 128, 128)
    assert tensor.device.type == "cpu"


def test_image_metrics_normalize(image_metrics: ImageMetrics):
    """Test the normalize method of ImageMetrics."""
    tensor = torch.rand(32, 128, 128)
    normalized = image_metrics.normalize(tensor)
    assert normalized.min() == 0
    assert normalized.max() == 1


def test_image_metrics_mse(image_metrics: ImageMetrics):
    """Test the mean squared error (MSE) calculation of ImageMetrics."""
    image = torch.rand(32, 128, 128)
    target = torch.rand(32, 128, 128)
    image_metrics.set_image(image)
    image_metrics.set_target(target)
    mse = image_metrics.mse
    assert isinstance(mse, float)
    assert 0 <= mse <= 1
