"""Verify all-detector self-splitting, context use and independent photon normalization."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.ssunet.models.confocal_gap import ConfocalGAPUNet
from src.ssunet.models.confocal_multichannel import (
    MultichannelConfocalGAP,
    channel_expected_counts,
    channel_log_probabilities,
    multichannel_gap_loss,
    split_all_channels,
)
from src.ssunet.softmax_fix import fixed_confocal_gap_loss
from src.tools.confocal_survey import _StedReader


@pytest.mark.parametrize("dimensions", [3, 4])
def test_all_channel_network_backpropagates_at_odd_shapes(dimensions):
    """Both physical channels have output logits and gradients in XYZ/XYZT modes."""
    torch.set_num_threads(1)
    model = MultichannelConfocalGAP(
        channels=2,
        dimensions=dimensions,
        base_channels=2,
        depth=2,
        encoding_levels=3,
        backend="torch",
    )
    counts = torch.arange(2 * 3 * 3 * 5 * 7).reshape(1, 2, 3, 3, 5, 7) % 4
    logits = model(counts)
    assert logits.shape == counts.shape
    loss = multichannel_gap_loss(logits, torch.ones_like(counts), counts)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.testing.assert_close(
        channel_log_probabilities(logits).exp().flatten(2).sum(-1),
        torch.ones(1, 2),
        atol=1e-6,
        rtol=0,
    )


@pytest.mark.parametrize("dimensions", [3, 4])
def test_one_channel_wrapper_matches_archived_network_exactly(dimensions):
    """Generic channel support retains the original C=1 network's numerical behavior."""
    torch.set_num_threads(1)
    original = ConfocalGAPUNet(
        dimensions=dimensions, base_channels=2, depth=2, encoding_levels=3, backend="torch"
    )
    generic = MultichannelConfocalGAP(
        channels=1,
        dimensions=dimensions,
        base_channels=2,
        depth=2,
        encoding_levels=3,
        backend="torch",
    )
    generic.network.load_state_dict(original.state_dict())
    x = torch.arange(3 * 3 * 5 * 7).reshape(1, 1, 3, 3, 5, 7) % 4
    torch.testing.assert_close(generic(x), original(x), rtol=0, atol=0)


def test_each_detector_has_its_own_offset_and_probability_mass():
    """Bright channels or arbitrary logit offsets cannot steal another channel's mass."""
    logits = torch.randn(2, 2, 3, 2, 4, 5, dtype=torch.float64)
    actual = channel_log_probabilities(logits)
    shifted = (
        logits
        + torch.tensor([1000.0, -1000.0], dtype=torch.float64)[None, :, None, None, None, None]
    )
    torch.testing.assert_close(actual, channel_log_probabilities(shifted), atol=2e-12, rtol=0)
    torch.testing.assert_close(
        actual.exp().flatten(2).sum(-1), torch.ones(2, 2, dtype=torch.float64)
    )


def test_channel_loss_is_mean_of_original_gap_losses_and_gradients():
    """Each detector's empirical counts define its own target distribution."""
    logits = torch.randn(1, 2, 2, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    target = torch.arange(logits.numel()).reshape_as(logits) % 5
    inputs = torch.ones_like(target)
    actual = multichannel_gap_loss(logits, target, inputs)
    expected = torch.stack(
        [
            fixed_confocal_gap_loss(
                logits[:, c : c + 1], target[:, c : c + 1], inputs[:, c : c + 1]
            )
            for c in range(2)
        ]
    ).mean()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    ga = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(ga, ge, atol=0, rtol=0)


def test_all_channel_splitting_conserves_counts_and_registration():
    """All detectors share crop coordinates, with independent disjoint photon splits."""
    base = np.arange(4 * 5 * 6 * 7, dtype=np.uint8).reshape(4, 5, 6, 7) % 4
    counts = np.stack([base, base * 3])
    a, b = split_all_channels(counts, [3, 3, 4, 5], seed=5, index=6, batch_size=2)
    assert a.shape == (2, 2, 3, 3, 4, 5)
    torch.testing.assert_close((a + b)[:, 1], (a + b)[:, 0] * 3)
    again = split_all_channels(counts, [3, 3, 4, 5], seed=5, index=6, batch_size=2)
    torch.testing.assert_close(a, again[0], rtol=0, atol=0)


def test_joint_model_uses_the_other_retained_channel_as_context():
    """Using all detectors changes useful context without feeding any target photons."""
    torch.set_num_threads(1)
    torch.manual_seed(9)
    model = MultichannelConfocalGAP(
        channels=2, dimensions=4, base_channels=2, depth=2, backend="torch"
    )
    x = torch.ones(1, 2, 3, 3, 4, 5, dtype=torch.int64)
    y = x.clone()
    y[:, 1] = 3
    assert float((model(x)[:, 0] - model(y)[:, 0]).abs().max().detach()) > 1e-6


@pytest.mark.parametrize("slow_axis", [2, 3])
def test_sted_tiff_page_order_matches_physical_zyx(slow_axis):
    """XZ-plane scans must not silently swap physical Y/Z or time."""
    values = np.arange(3 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 4, 5, 6)
    planes = (
        values.reshape(-1, 5, 6)
        if slow_axis == 3
        else values.transpose(0, 2, 1, 3).reshape(-1, 4, 6)
    )
    pages = [SimpleNamespace(asarray=lambda a=a: a) for a in planes]
    reader = _StedReader(pages, values.shape, slow_axis)
    for time, z in [(0, 0), (1, 3), (2, 1), (0, 2)]:
        np.testing.assert_array_equal(reader.frame(0, time, z), values[time, z])


def test_count_calibration_uses_every_detectors_own_exposure():
    """A bright detector cannot inflate another detector's expected count map."""
    observed = torch.ones(1, 2, 2, 2, 3, 4, dtype=torch.int64)
    observed[:, 1] *= 7
    logits = torch.zeros_like(observed, dtype=torch.float32)
    expected = channel_expected_counts(logits, observed)
    torch.testing.assert_close(expected, observed.float())
    observed[:, 0] = 0
    assert not channel_expected_counts(logits, observed)[:, 0].any()


@pytest.mark.parametrize("kind", ["float", "complex", "boolean", "negative"])
def test_count_calibration_rejects_values_that_are_not_photon_counts(kind):
    """Display calibration requires measured nonnegative counts, without gain scaling."""
    observed = torch.ones(1, 2, 2, 2, 3, 4, dtype=torch.int64)
    logits = torch.zeros_like(observed, dtype=torch.float32)
    if kind == "negative":
        observed[:, 0] = -1
    else:
        observed = observed.to(
            {"float": torch.float32, "complex": torch.complex64, "boolean": torch.bool}[kind]
        )
    with pytest.raises(ValueError, match="raw nonnegative integer"):
        channel_expected_counts(logits, observed)


def test_all_channel_cache_preserves_crop_channel_order_and_rejects_corruption(
    tmp_path, monkeypatch
):
    """Both detector values stay co-registered and byte-exact through network caching."""
    from src.ssunet.datasets import confocal_channels

    raw = np.arange(5 * 4 * 2 * 8 * 10, dtype=np.uint8).reshape(5, 4, 2, 8, 10) % 13
    path = tmp_path / "all.lif"
    path.write_bytes(b"header" + raw.tobytes())
    image = SimpleNamespace(
        dims=SimpleNamespace(t=5, z=4, y=8, x=10),
        channels=2,
        display_dims=(1, 2),
        n_mosaic=1,
        offsets=(6, raw.size),
        channel_as_second_dim=False,
        get_frame=lambda t, z, c: raw[t, z, c],
    )
    identity = {"source": "test", "active_detectors": [{"Name": "A"}, {"Name": "B"}]}
    monkeypatch.setattr(confocal_channels, "_metadata", lambda _: (image, identity))
    source = {"path": str(path), "crop_tzyx": [[1, 5], [1, 4], [2, 7], [3, 9]], "channels": "all"}
    cached, manifest = confocal_channels.prepare_all_channels(source, tmp_path / "counts.npy")
    np.testing.assert_array_equal(cached, raw[1:5, 1:4, :, 2:7, 3:9].transpose(2, 0, 1, 3, 4))
    assert manifest["channel_indices"] == [0, 1]
    changed = np.load(tmp_path / "counts.npy", mmap_mode="r+")
    changed[1, 0, 0, 0, 0] ^= 1
    changed.flush()
    with pytest.raises(ValueError, match="identity or content"):
        confocal_channels.prepare_all_channels(source, tmp_path / "counts.npy")
