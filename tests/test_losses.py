"""Distribution and gradient regressions for photon prediction losses."""

import math

import pytest
import torch

from src.ssunet.distributions import logits_to_probabilities
from src.ssunet.losses import photon_loss, photon_loss_2d, photon_nll


@pytest.mark.parametrize(
    ("loss_function", "dims"),
    [(photon_loss, (-1, -2, -3, -4)), (photon_loss_2d, (-1, -2, -4))],
)
@pytest.mark.parametrize("masked", [False, True])
def test_photon_loss_matches_masked_offset_objective_and_gradients(loss_function, dims, masked):
    """Only included exponential weights contribute to the normalization offset."""
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(2, 2, 3, 4, 4, dtype=torch.float64, generator=generator)
    logits.requires_grad_()
    target = torch.rand(logits.shape, dtype=torch.float64, generator=generator)
    mask = (
        (torch.rand(logits.shape, generator=generator) > 0.3).to(logits.dtype) if masked else None
    )
    old_target = target if mask is None else target * mask
    energy = logits.exp() if mask is None else logits.exp() * mask
    expected = (
        -(logits * old_target).mean(dim=dims, keepdim=True)
        + energy.mean(dim=dims, keepdim=True).log() * old_target.mean(dim=dims, keepdim=True)
    ).mean()
    actual = loss_function(logits, target, mask)
    expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, logits)[0]

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-12)
    if mask is not None:
        assert torch.count_nonzero(actual_gradient[mask == 0]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("loss_function", [photon_loss, photon_loss_2d])
def test_photon_loss_handles_extreme_log_weights(dtype, loss_function):
    """Finite large log weights retain finite losses and gradients without clipping."""
    logits = torch.tensor([-1000.0, 1000.0, -900.0, 900.0], dtype=dtype)
    logits = logits.reshape(1, 1, 1, 2, 2).requires_grad_()
    target = torch.ones_like(logits)
    loss = loss_function(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    if dtype in (torch.float16, torch.bfloat16):
        assert loss.dtype == torch.float32


def test_empty_photon_targets_have_zero_loss_and_gradient():
    """A photon-free target carries no training evidence and must remain finite."""
    logits = torch.randn(2, 1, 2, 4, 4, requires_grad=True)
    target = torch.zeros_like(logits)
    loss = photon_loss(logits, target)
    loss.backward()

    assert loss == 0
    assert torch.count_nonzero(logits.grad) == 0
    assert photon_nll(logits.detach(), target) == 0


@pytest.mark.parametrize("masked", [False, True])
def test_probabilities_and_loss_ignore_logit_offsets(masked):
    """The binary mask cannot introduce an absolute-logit anchor into the offset."""
    generator = torch.Generator().manual_seed(29)
    logits = torch.randn(2, 2, 3, 4, 4, dtype=torch.float64, generator=generator)
    target = torch.rand(logits.shape, dtype=torch.float64, generator=generator)
    mask = torch.rand(logits.shape, generator=generator) > 0.3 if masked else None
    offsets = torch.tensor([-1000.0, 1000.0], dtype=logits.dtype).reshape(2, 1, 1, 1, 1)
    probabilities = logits_to_probabilities(logits, mask=mask)
    shifted = logits_to_probabilities(logits + offsets, mask=mask)

    torch.testing.assert_close(probabilities.sum(dim=(1, 2, 3, 4)), torch.ones(2).double())
    torch.testing.assert_close(probabilities, shifted, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        photon_loss(logits, target, mask), photon_loss(logits + offsets, target, mask)
    )
    assert probabilities.dtype == torch.float64


def test_2d_probabilities_normalize_each_temporal_slice_separately():
    """The 2D diagnostic retains the existing per-time channel/spatial axes."""
    logits = torch.arange(24, dtype=torch.float16).reshape(1, 2, 3, 2, 2)
    probabilities = logits_to_probabilities(logits, dim=(-1, -2, -4))

    assert probabilities.dtype == torch.float32
    torch.testing.assert_close(probabilities.sum(dim=(-1, -2, -4)), torch.ones(1, 3))


def test_photon_nll_normalizes_events_and_ignores_empty_samples():
    """Empty windows and varying target counts do not dilute the reported score."""
    logits = torch.zeros(2, 1, 1, 2, 2)
    target = torch.zeros_like(logits)
    target[1, 0, 0, 0, 0] = 7

    torch.testing.assert_close(photon_nll(logits, target), torch.tensor(math.log(4)))


def test_photon_nll_excludes_masked_exponential_mass():
    """Three included uniform locations normalize to 1/3, regardless of excluded logits."""
    logits = torch.tensor([100.0, 0.0, 0.0, 0.0]).reshape(1, 1, 1, 2, 2)
    target = torch.ones_like(logits)
    mask = torch.tensor([0.0, 1.0, 1.0, 1.0]).reshape_as(logits)

    torch.testing.assert_close(photon_nll(logits, target, mask), torch.tensor(math.log(3)))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("loss_function", [photon_loss, photon_loss_2d])
def test_fully_masked_photon_loss_has_zero_value_and_gradient(dtype, loss_function):
    """No supported target means zero evidence, including in mixed precision."""
    logits = torch.tensor([-1000, 1000, -900, 900], dtype=dtype)
    logits = logits.reshape(1, 1, 1, 2, 2).requires_grad_()
    target, mask = torch.ones_like(logits), torch.zeros_like(logits)
    loss = loss_function(logits, target, mask)
    loss.backward()
    assert loss == 0
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == 0
    assert photon_nll(logits.detach(), target, mask) == 0


def test_masked_probabilities_exclude_mass_and_handle_empty_frames():
    """Per-frame support excludes a bright input hit and safely skips an all-masked frame."""
    logits = torch.tensor([1000.0, math.log(2), math.log(3), math.log(5)] * 2)
    logits = logits.reshape(1, 1, 2, 2, 2).requires_grad_()
    mask = torch.tensor([False, True, True, True, False, False, False, False]).reshape_as(logits)
    q = logits_to_probabilities(logits, dim=(-1, -2, -4), mask=mask)
    torch.testing.assert_close(q[0, 0, 0].flatten(), torch.tensor([0.0, 0.2, 0.3, 0.5]))
    assert torch.count_nonzero(q[0, 0, 1]) == 0
    q[0, 0, 0, 1, 1].backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[~mask]) == 0
