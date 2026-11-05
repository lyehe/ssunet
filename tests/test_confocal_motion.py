"""Check continuous time reconstruction against direct convolution and static controls."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.ssunet.confocal_motion import (
    normalized_motion_probabilities,
    split_motion_pool,
    temporal_control,
    temporal_logits,
)
from src.tools.confocal_motion_report import bulk_shift


class TemporalReference(nn.Module):
    """Two known temporal kernels give an independent direct-convolution reference."""

    def __init__(self):
        super().__init__()
        self.channels = 2
        self.network = SimpleNamespace(encoder=[None], decoder=[])
        self.layers = nn.Sequential(
            nn.Conv1d(2, 2, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv1d(2, 2, 3, padding=1, bias=False),
        )
        for parameter in self.parameters():
            nn.init.constant_(parameter, 0.2)

    def forward(self, counts):
        """Convolve time directly at every spatial location."""
        n, c, t, z, y, x = counts.shape
        values = counts.float().permute(0, 3, 4, 5, 1, 2).reshape(n * z * y * x, c, t)
        return self.layers(values).reshape(n, z, y, x, c, t).permute(0, 4, 5, 1, 2, 3)


def _counts():
    return (np.arange(2 * 12 * 2 * 3 * 3).reshape(2, 12, 2, 3, 3) % 7).astype(np.uint8)


def test_halo_stitch_matches_one_direct_forward_over_the_continuous_clip():
    """Chunk boundaries cannot introduce new intensity or position discontinuities."""
    torch.set_num_threads(1)
    model = TemporalReference().eval()
    counts = _counts()
    actual = temporal_logits(model, counts, frames=8, halo=2, core=3, device=torch.device("cpu"))
    with torch.inference_mode():
        expected = model(torch.from_numpy(counts)[None])[0, :, 2:10].numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-6)


def test_static_control_detects_unprotected_edges_and_removes_them_with_real_halos():
    """Apparent temporal change from zero padding must not be interpreted as movement."""
    control = temporal_control(
        TemporalReference().eval(), _counts(), halo=2, core=4, device=torch.device("cpu")
    )
    assert control["status"] == "passed"
    assert control["overlapping_core_max_logit_difference"] < 2e-6
    assert control["static_input_halo_core_max_logit_difference"] < 2e-6
    assert control["static_input_without_halo_max_shape_logit_difference"] > 1e-5


def test_inference_rejects_temporal_context_shorter_than_the_network_receptive_field():
    """Do not silently reconstruct motion using padded temporal boundary values."""
    with pytest.raises(ValueError, match="real temporal halo"):
        temporal_logits(
            TemporalReference(),
            _counts()[:, :10],
            frames=8,
            halo=1,
            core=4,
            device=torch.device("cpu"),
        )


def test_motion_photon_partition_is_disjoint_and_deterministic_across_windows():
    """All chunks must reuse one fixed photon split of the continuous acquisition."""
    counts = _counts()
    a, b = split_motion_pool(counts, seed=7, first_time=242)
    np.testing.assert_array_equal(a + b, counts)
    aa, bb = split_motion_pool(counts, seed=7, first_time=242)
    np.testing.assert_array_equal(a, aa)
    np.testing.assert_array_equal(b, bb)


def test_motion_pdfs_normalize_over_the_full_clip_independently_per_detector():
    """Global normalization preserves one unit of mass despite temporal chunking."""
    logits = np.linspace(-10, 10, 2 * 8 * 2 * 3 * 3, dtype=np.float32).reshape(2, 8, 2, 3, 3)
    q = normalized_motion_probabilities(logits, torch.device("cpu"))
    np.testing.assert_allclose(q.sum(axis=(1, 2, 3, 4), dtype=np.float64), 1, atol=1e-7)
    shifted = logits + np.array([-50, 50], dtype=np.float32)[:, None, None, None, None]
    np.testing.assert_allclose(
        normalized_motion_probabilities(shifted, torch.device("cpu")), q, rtol=1e-5, atol=1e-7
    )


def test_bulk_registration_reports_the_known_displacement_direction():
    """An alignment shift has the opposite sign to the observed feature displacement."""
    y, x = np.mgrid[:64, :64]
    reference = np.exp(-((y - 22) ** 2 + (x - 25) ** 2) / 20)
    reference += 0.4 * np.exp(-((y - 39) ** 2 + (x - 41) ** 2) / 15)
    moving = np.roll(np.roll(reference, 3, axis=0), -2, axis=1)
    actual = bulk_shift(reference, moving)
    assert actual["dx_pixels"] == pytest.approx(-2, abs=0.1)
    assert actual["dy_pixels"] == pytest.approx(3, abs=0.1)


def test_bulk_registration_does_not_turn_uniform_brightness_change_into_motion():
    """Global fading alone must retain zero estimated displacement."""
    y, x = np.mgrid[:64, :64]
    reference = np.exp(-((y - 22) ** 2 + (x - 25) ** 2) / 20)
    actual = bulk_shift(reference, reference * 3 + 5)
    assert actual["dx_pixels"] == pytest.approx(0, abs=0.1)
    assert actual["dy_pixels"] == pytest.approx(0, abs=0.1)
