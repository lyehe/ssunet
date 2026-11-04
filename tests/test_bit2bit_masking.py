"""Exercise normalization masking through the actual Bit2Bit model loss hooks."""

import math

import pytest
import torch

from src.ssunet.configs.model_config import ModelConfig
from src.ssunet.models.bit2bit import Bit2Bit


@pytest.mark.parametrize("loss_function", ["photon", "photon_2d"])
def test_bit2bit_model_masks_the_normalization_offset(loss_function):
    """Both model training and diagnostic hooks exclude exp(input-hit logits)."""
    model = Bit2Bit(ModelConfig(depth=2, start_filts=8, loss_function=loss_function, masked=True))
    logits = torch.tensor([1000.0, 0.0, 0.0, 0.0]).reshape(1, 1, 1, 2, 2).requires_grad_()
    inputs = torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape_as(logits)
    target = torch.tensor([0.0, 4.0, 0.0, 0.0]).reshape_as(logits)
    loss = model._batch_loss(logits, target, inputs)
    torch.testing.assert_close(loss, torch.tensor(math.log(3 / 4)))
    torch.testing.assert_close(
        model._photon_score(logits, target, inputs), torch.tensor(math.log(3))
    )
    loss.backward()
    assert logits.grad[inputs > 0].abs().sum() == 0


def test_bit2bit_inference_keeps_the_full_observation_support(monkeypatch):
    """Training's dependence mask must not erase observed-photon locations at inference."""
    model = Bit2Bit(ModelConfig(depth=2, start_filts=8, masked=True))
    inputs = torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 1, 2, 2)
    monkeypatch.setattr(model, "forward", lambda data: torch.zeros_like(data))
    q = model.predict_distribution(inputs)
    torch.testing.assert_close(q, torch.full_like(inputs, 0.25))
