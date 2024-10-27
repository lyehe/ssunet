import pytest
import torch

from src.ssunet.exceptions import InvalidUpModeError
from src.ssunet.models import Bit2Bit
from ssunet.configs.configs import ModelConfig


def test_ssunet_initialization():
    """Test the initialization of SSUnet model."""
    model_config = ModelConfig(channels=1, depth=5, start_filts=32, up_mode="transpose")
    model = Bit2Bit(model_config)
    assert isinstance(model, Bit2Bit)
    assert next(model.parameters()).device.type == "cpu"


def test_ssunet_forward_pass():
    """Test the forward pass of SSUnet model."""
    model_config = ModelConfig(channels=1, depth=5, start_filts=32, up_mode="transpose")
    model = Bit2Bit(model_config)

    input_tensor = torch.randn(1, 1, 32, 128, 128, device="cpu")
    output = model(input_tensor)

    assert output.device.type == "cpu"
    assert output.shape == input_tensor.shape


def test_ssunet_invalid_up_mode():
    """Test SSUnet initialization with invalid up_mode."""
    model_config = ModelConfig(channels=1, depth=5, start_filts=32, up_mode="invalid_mode")
    with pytest.raises(InvalidUpModeError):
        Bit2Bit(model_config)


def test_ssunet_different_input_sizes():
    """Test SSUnet forward pass with different input sizes."""
    model_config = ModelConfig(channels=1, depth=5, start_filts=32, up_mode="transpose")
    model = Bit2Bit(model_config)

    input_sizes = [(1, 1, 32, 128, 128), (1, 1, 16, 64, 64), (2, 1, 32, 256, 256)]

    for size in input_sizes:
        input_tensor = torch.randn(*size, device="cpu")
        output = model(input_tensor)
        assert output.device.type == "cpu"
        assert output.shape == input_tensor.shape
