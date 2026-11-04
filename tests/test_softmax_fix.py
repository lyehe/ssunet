"""Check Barron's collapse example, analytic gradients and unchanged GAP weighting."""

import math

import pytest
import torch

from src.ssunet.models.confocal_gap import confocal_gap_loss
from src.ssunet.softmax_fix import fixed_confocal_gap_loss, fixed_log_softmax, fixed_softmax


@pytest.mark.parametrize(("dtype", "peak"), [(torch.float32, 9.0), (torch.float64, 20.0)])
def test_high_confidence_loss_and_dominant_gradient_survive_precision_cliff(dtype, peak):
    """Compare tiny nonzero losses/gradients with an analytic two-tail reference."""
    logits = torch.tensor([peak, -peak, -peak], dtype=dtype, requires_grad=True)
    loss = -fixed_log_softmax(logits)[0]
    gradient = torch.autograd.grad(loss, logits)[0]
    tail = 2 * math.exp(-2 * peak)
    assert loss > 0
    assert gradient[0] < 0
    assert loss.item() == pytest.approx(math.log1p(tail), rel=3e-6)
    assert gradient[0].item() == pytest.approx(-tail / (1 + tail), rel=3e-6)
    assert gradient[1].item() == pytest.approx(tail / (2 * (1 + tail)), rel=3e-6)
    assert abs(gradient.sum().item()) < abs(gradient[0].item()) * 1e-6


def test_softmax_dominant_derivative_remains_nonzero_even_if_probability_rounds_to_one():
    """The exponential of fixed log-softmax preserves the corrected backward path."""
    logits = torch.tensor([9.0, -9.0, -9.0], requires_grad=True)
    probability = fixed_softmax(logits)[0]
    gradient = torch.autograd.grad(probability, logits)[0]
    assert gradient[0] > 0
    torch.testing.assert_close(gradient[0], -gradient[1:].sum(), rtol=1e-6, atol=0)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_fixed_log_softmax_matches_regular_precision_values_and_gradients(dim):
    """Ordinary logits retain the same probability model, axis semantics and Jacobian."""
    torch.manual_seed(17)
    logits = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    weights = torch.randn_like(logits)
    actual = fixed_log_softmax(logits, dim=dim)
    expected = logits.log_softmax(dim)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    a = torch.autograd.grad((actual * weights).sum(), logits, retain_graph=True)[0]
    b = torch.autograd.grad((expected * weights).sum(), logits)[0]
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_fixed_log_softmax_passes_double_precision_gradcheck():
    """Exercise automatic differentiation away from argmax ties."""
    logits = torch.tensor([[0.2, 1.1, -0.7, 0.5]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fixed_log_softmax, (logits,))


def test_singleton_and_tied_logits_have_finite_correct_gradients():
    """A single supported location is deterministic; equal maxima need no special bias."""
    one = torch.tensor([[100.0]], requires_grad=True)
    fixed_log_softmax(one).sum().backward()
    assert torch.count_nonzero(one.grad) == 0
    tied = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)
    result = fixed_log_softmax(tied)
    torch.testing.assert_close(result, torch.full_like(tied, -math.log(4)))
    torch.testing.assert_close(
        torch.autograd.grad(-result[:, 0].sum(), tied)[0],
        torch.tensor([[-0.75, 0.25, 0.25, 0.25]] * 2, dtype=torch.float64),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_inputs_use_float32_accumulation(dtype):
    """Keep loss computation outside low-precision probability arithmetic."""
    logits = torch.tensor([[9.0, -9.0, -9.0]], dtype=dtype)
    actual = fixed_log_softmax(logits)
    assert actual.dtype == torch.float32
    assert actual[0, 0] < 0


def test_fixed_gap_preserves_published_target_weighting_and_empty_target_behavior():
    """Changing numerical normalization does not replace GAP's empirical count targets."""
    torch.manual_seed(11)
    logits = torch.randn(2, 1, 2, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    target = torch.arange(logits.numel()).reshape(logits.shape) % 7
    target[0] = 0
    retained = torch.ones_like(target)
    actual = fixed_confocal_gap_loss(logits, target, retained)
    expected = confocal_gap_loss(logits, target, retained)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    a = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    b = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(a[0]) == 0


def test_fixed_gap_retains_tiny_gradient_on_a_dominant_observed_photon_bin():
    """Keep the stable gradient through the added constant in GAP's unit-mean loss."""
    logits = torch.tensor([9.0, -9.0, -9.0]).reshape(1, 1, 1, 1, 1, 3).requires_grad_()
    target = torch.tensor([1, 0, 0]).reshape_as(logits)
    gradient = torch.autograd.grad(
        fixed_confocal_gap_loss(logits, target, torch.zeros_like(target)), logits
    )[0].flatten()
    assert gradient[0] < 0
    torch.testing.assert_close(gradient[0], -gradient[1:].sum(), rtol=1e-6, atol=0)
