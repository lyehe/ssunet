"""Verify motion-control distributions with known kernels and analytic photon scores."""

import math

import numpy as np
import pytest
from scipy.ndimage import convolve1d

from src.ssunet.confocal_motion_comparison import (
    factorized_motion_probability,
    gaussian_motion_probability,
    photon_log_score,
)


def test_gaussian_control_matches_explicit_separable_kernel_on_the_same_photon_inputs():
    """Check real-context cropping and the normalized mixture against an independent kernel."""
    first = np.zeros((13, 5, 7, 8), dtype=np.uint8)
    first[4, 2, 3, 4] = 5
    second = np.zeros_like(first)
    second[8, 1, 4, 3] = 3
    sigma = (1.0, 0.5, 1.5, 1.5)
    reference = np.zeros((7, 5, 7, 8), dtype=np.float64)
    for counts in (first, second):
        values = counts.astype(np.float64)
        for axis, width in enumerate(sigma):
            radius = int(3 * width + 0.5)
            distance = np.arange(-radius, radius + 1)
            kernel = np.exp(-0.5 * (distance / width) ** 2)
            values = convolve1d(values, kernel / kernel.sum(), axis=axis, mode="reflect")
        values = values[3:10]
        reference += 0.5 * (0.99 * values / values.sum() + 0.01 / values.size)
    actual = gaussian_motion_probability(first, second, sigma=sigma, halo=3, frames=7)
    np.testing.assert_allclose(actual, reference, rtol=4e-7, atol=1e-10)
    assert actual.sum(dtype=np.float64) == pytest.approx(1, abs=2e-8)


def test_spatial_only_control_cannot_transfer_signal_between_times():
    """The spatial baseline leaves empty times at the declared uniform floor."""
    counts = np.zeros((5, 3, 5, 5), dtype=np.uint8)
    counts[2, 1, 2, 2] = 1
    q = gaussian_motion_probability(counts, counts, sigma=(0, 0.5, 1, 1), halo=0, frames=5)
    np.testing.assert_allclose(q[[0, 1, 3, 4]], 0.01 / q.size)
    assert q[2].sum() == pytest.approx(0.992, abs=1e-7)


def test_empty_photon_inputs_produce_uniform_probability_without_nan():
    """No-count data needs a valid predictive distribution, rather than division by zero."""
    counts = np.zeros((7, 2, 3, 3), dtype=np.uint8)
    q = gaussian_motion_probability(counts, counts, sigma=(1, 0.5, 1, 1), halo=3, frames=1)
    np.testing.assert_allclose(q, 1 / q.size)
    score = photon_log_score(q, counts[:1])
    assert score["photons"] == 0
    assert score["nats_per_photon"] is None


def test_gaussian_control_rejects_insufficient_real_context():
    """An artificial temporal boundary cannot silently replace requested context."""
    counts = np.zeros((5, 2, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="real temporal halo"):
        gaussian_motion_probability(counts, counts, sigma=(1, 0.5, 1, 1), halo=2, frames=1)


def test_frozen_shape_control_preserves_both_marginals_and_removes_interactions():
    """Brightness over time is unchanged, while spatial shape becomes identical at every time."""
    q = np.arange(1, 25, dtype=np.float64).reshape(4, 2, 3, 1)
    q /= q.sum()
    frozen = factorized_motion_probability(q)
    temporal = q.sum(axis=(1, 2, 3))
    np.testing.assert_allclose(frozen.sum(axis=(1, 2, 3)), temporal)
    np.testing.assert_allclose(frozen.sum(axis=0), q.sum(axis=0))
    for time in range(4):
        np.testing.assert_allclose(frozen[time] / temporal[time], q.sum(axis=0))


def test_photon_score_matches_analytic_multinomial_loss_and_normalizes_pdf_mass():
    """Use known integer photon multiplicities, not a squared-error image metric."""
    counts = np.array([3, 1, 0, 2], dtype=np.uint8).reshape(2, 1, 1, 2)
    q = np.array([0.5, 0.25, 0.125, 0.125]).reshape(counts.shape)
    expected = -(3 * math.log(0.5) + math.log(0.25) + 2 * math.log(0.125))
    result = photon_log_score(q * 7, counts, block=1)
    assert result["negative_log_likelihood"] == pytest.approx(expected, abs=1e-12)
    assert result["nats_per_photon"] == pytest.approx(expected / 6)
    assert [item["photons"] for item in result["blocks"]] == [4, 2]


def test_known_time_dependent_photon_distribution_beats_its_frozen_shape():
    """Software-only moving point probabilities establish the sign of the dynamic score gain."""
    counts = np.array([9, 1, 1, 9], dtype=np.uint8).reshape(2, 1, 1, 2)
    q = counts.astype(np.float64) / counts.sum()
    dynamic = photon_log_score(q, counts)["nats_per_photon"]
    static = photon_log_score(factorized_motion_probability(q), counts)["nats_per_photon"]
    assert static - dynamic == pytest.approx(
        math.log(2) + 0.9 * math.log(0.9) + 0.1 * math.log(0.1)
    )


def test_photon_score_rejects_zero_probability_at_an_observed_photon():
    """Invalid predictions must not become finite through hidden numerical clipping."""
    counts = np.array([1, 0], dtype=np.uint8).reshape(1, 1, 1, 2)
    q = np.array([0, 1], dtype=np.float32).reshape(counts.shape)
    with pytest.raises(ValueError, match="strictly positive"):
        photon_log_score(q, counts)
