"""Check photon-method routing, GAP weighting and Bit2Bit's masked offset."""

import math

import pytest
import torch

from src.ssunet.photon_methods import (
    method_for_observation,
    paper_photon_loss,
    sinusoidal_photon_features,
)
from src.ssunet.photon_reference import photon_count_log_probabilities, photon_count_nll


@pytest.mark.parametrize(("kind", "method"), [("one_bit", "bit2bit"), ("photon_counts", "gap")])
def test_method_follows_acquisition_contract(kind, method):
    """A sparse count batch does not change the declared observation family."""
    assert method_for_observation(kind) == method
    with pytest.raises(ValueError, match="Declare"):
        method_for_observation("infer_from_this_batch")


@pytest.mark.parametrize("kind", ["photon_counts", "one_bit"])
def test_target_normalization_and_masked_offset_gradients(kind):
    """Keep GAP's published weighting and explicitly exclude the binary offset mass."""
    torch.manual_seed(14)
    logits = (torch.randn(2, 2, 3, dtype=torch.float64) / 3).requires_grad_()
    inputs = torch.tensor([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]])
    target = torch.tensor([[[0, 1, 0], [1, 0, 1]], [[1, 1, 0], [0, 0, 1]]])
    if kind == "photon_counts":
        target = target * 7
    live = torch.ones(2, 2, dtype=torch.bool)
    epsilon = 1e-8
    normalized = target.double() / (target.double().mean((1, 2), keepdim=True) + epsilon)
    energy = logits.exp()
    if kind == "one_bit":
        energy = energy * (inputs == 0)
    expected = (
        -(logits * normalized).mean((1, 2)) + energy.mean((1, 2)).log() * normalized.mean((1, 2))
    ).mean()
    actual = paper_photon_loss(logits, target, live, inputs, observation_kind=kind)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    if kind == "one_bit":
        assert torch.count_nonzero(actual_gradient[inputs > 0]) == 0


def test_target_weighting_changes_event_weights_not_probability_normalization():
    """A 100-photon target has 100x weight only in the archived raw-count objective."""
    logits = torch.zeros(2, 2, 2, dtype=torch.float64, requires_grad=True)
    target = torch.zeros(2, 2, 2, dtype=torch.int64)
    target[:, 0, 0] = torch.tensor([1, 100])
    live = torch.ones(2, 2, dtype=torch.bool)
    counts = torch.zeros_like(target)
    raw = torch.autograd.grad(photon_count_nll(logits, target, live), logits)[0]
    normalized = torch.autograd.grad(
        paper_photon_loss(logits, target, live, counts, observation_kind="photon_counts"), logits
    )[0]
    torch.testing.assert_close(raw[1], raw[0] * 100)
    torch.testing.assert_close(normalized[1], normalized[0], rtol=1e-6, atol=1e-8)
    q = photon_count_log_probabilities(logits, live).exp()
    torch.testing.assert_close(q.sum((1, 2)), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(q.sum(-1), torch.full((2, 2), 0.5, dtype=torch.float64))


def test_counts_can_share_a_bin_but_binary_splits_cannot():
    """Applying a Bit2Bit hit mask to a multi-photon bin is rejected explicitly."""
    logits = torch.zeros(1, 2, 3, requires_grad=True)
    counts, target = (
        torch.zeros(1, 2, 3, dtype=torch.int64),
        torch.zeros(1, 2, 3, dtype=torch.int64),
    )
    counts[0, 0, 1], target[0, 0, 1] = 2, 3
    live = torch.ones(2, dtype=torch.bool)
    assert torch.isfinite(
        paper_photon_loss(logits, target, live, counts, observation_kind="photon_counts")
    )
    with pytest.raises(ValueError, match="one-bit"):
        paper_photon_loss(logits, target, live, counts, observation_kind="one_bit")


@pytest.mark.parametrize("kind", ["photon_counts", "one_bit"])
def test_empty_targets_and_dead_support_are_safe(kind):
    """An empty target has zero gradient; dead detectors receive zero mass."""
    logits = torch.tensor([[[0.0, 10000.0], [10000.0, 10000.0]]], requires_grad=True)
    live = torch.tensor([True, False])
    zeros = torch.zeros_like(logits, dtype=torch.int64)
    loss = paper_photon_loss(logits, zeros, live, zeros, observation_kind=kind)
    assert loss == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0
    assert photon_count_log_probabilities(logits, live).exp()[:, 1].sum() == 0


def test_sine_encoding_retains_brightness_without_unit_mean_input_scaling():
    """Counts at 1 and 100 map to different published multiscale sine features."""
    counts = torch.tensor([0, 1, 100])
    encoded = sinusoidal_photon_features(counts)
    assert encoded.shape == (3, 10)
    assert encoded[0].abs().sum() == 0
    assert encoded[1, 0] == pytest.approx(math.sin(1), abs=1e-7)
    assert encoded[2, 2] == pytest.approx(math.sin(1), abs=1e-7)
    assert not torch.equal(encoded[1], encoded[2])


def test_bit2bit_masks_the_offset_without_exp_zero_phantom_mass():
    """The loss offset contains three exponentials, not three exponentials plus one."""
    logits = torch.zeros(1, 1, 4, dtype=torch.float64, requires_grad=True)
    inputs = torch.tensor([[[1, 0, 0, 0]]])
    target = torch.tensor([[[0, 1, 0, 0]]])
    live = torch.tensor([True])
    loss = paper_photon_loss(logits, target, live, inputs, observation_kind="one_bit")
    assert loss.item() == pytest.approx(math.log(3 / 4) / (1 + 4e-8), abs=1e-12)
    gradient = torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(
        gradient,
        torch.tensor([[[0.0, -2 / 3, 1 / 3, 1 / 3]]], dtype=torch.float64) / (1 + 4e-8),
    )
    for offset in (-1000, 1000):
        shifted = paper_photon_loss(
            logits + offset, target, live, inputs, observation_kind="one_bit"
        )
        torch.testing.assert_close(shifted, loss)


def test_bit2bit_all_input_hits_and_dead_support_have_no_loss_or_gradient():
    """All-masked binary splits carry no target evidence, even in a mixed batch."""
    inputs = torch.tensor([[[1, 1], [0, 0]], [[0, 0], [0, 0]]])
    target = torch.zeros_like(inputs)
    target[1, 0, 0] = 1
    live = torch.tensor([True, False])
    logits = torch.tensor([[[1000.0, -1000.0], [999.0, -999.0]]] * 2, requires_grad=True)
    loss = paper_photon_loss(logits, target, live, inputs, observation_kind="one_bit")
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[0]) == 0
    assert torch.count_nonzero(logits.grad[:, 1]) == 0
