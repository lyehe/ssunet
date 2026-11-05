"""Verify inference units, coverage, count splitting, source I/O and restart safety."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.ssunet import confocal_inference as inference
from src.tools.confocal_infer import finalize


class Uniform(torch.nn.Module):
    """Predict a uniform distribution while retaining the requested tensor shape."""

    def forward(self, x):
        """Return logits with an arbitrary constant normalization offset."""
        return torch.full_like(x.float(), 1000)


@pytest.mark.parametrize("shape", [(2, 5, 11, 13), (1, 1, 2, 3)])
@pytest.mark.parametrize("microbatch", [1, 2])
def test_uniform_prediction_preserves_count_units_and_overlap_coverage(shape, microbatch):
    """Odd boundaries and shorter-than-patch axes reconstruct a constant count field."""
    raw = np.full(shape, 4, dtype=np.uint8)
    a = inference.photon_half(raw, first_time=0, seed=42)
    result = inference.reconstruct_core(
        Uniform(),
        raw,
        a,
        core_slice=slice(None),
        patch_zyx=(3, 7, 8),
        overlap_zyx=(1, 3, 4),
        device=torch.device("cpu"),
        microbatch_size=microbatch,
    )
    np.testing.assert_allclose(result, 4, rtol=2e-6)


def test_empty_region_stays_empty_without_inventing_count_mass():
    """No observed photons implies zero count-equivalent reconstructed intensity."""
    raw = np.zeros((2, 3, 8, 8), dtype=np.uint8)
    result = inference.reconstruct_core(
        Uniform(),
        raw,
        raw,
        core_slice=slice(1, 2),
        patch_zyx=(2, 6, 6),
        overlap_zyx=(1, 2, 2),
        device=torch.device("cpu"),
    )
    assert result.shape == (1, 3, 8, 8)
    assert not result.any()


def test_photon_halves_are_conserved_and_invariant_to_temporal_chunking():
    """Halo windows reuse the identical thinning of each real recorded volume."""
    raw = np.random.default_rng(1).integers(0, 7, (7, 3, 5, 8), dtype=np.uint8)
    full = inference.photon_half(raw, first_time=10, seed=123)
    partial = inference.photon_half(raw[2:5], first_time=12, seed=123)
    np.testing.assert_array_equal(full[2:5], partial)
    assert np.all(full <= raw)
    np.testing.assert_array_equal(full + (raw - full), raw)


@pytest.mark.parametrize("bad", ["dtype", "count"])
def test_reconstruction_rejects_changed_count_units(bad):
    """Floating intensities and invalid retained photons cannot silently enter GAP."""
    raw = np.ones((1, 2, 3, 4), dtype=np.uint8)
    a = raw.astype(np.float32) if bad == "dtype" else raw + 1
    with pytest.raises(ValueError, match="Retained counts"):
        inference.reconstruct_core(
            Uniform(),
            raw,
            a,
            core_slice=slice(None),
            patch_zyx=(2, 3, 4),
            overlap_zyx=(0, 0, 0),
            device=torch.device("cpu"),
        )


def test_bulk_cache_matches_channel_layout_and_rejects_corruption(tmp_path, monkeypatch):
    """Sequential full-volume LIF reads preserve all voxels of the selected detector."""
    data = np.arange(4 * 3 * 2 * 5 * 7, dtype=np.uint8).reshape(4, 3, 2, 5, 7) % 8
    source = tmp_path / "source.lif"
    source.write_bytes(b"header" + data.tobytes())
    image = SimpleNamespace(
        dims=SimpleNamespace(t=4, z=3, y=5, x=7),
        channels=2,
        display_dims=(1, 2),
        n_mosaic=1,
        bit_depth=(8, 8),
        offsets=(6, data.size),
        channel_as_second_dim=False,
        scale=(2, 2, -1, 1),
        get_frame=lambda t, z, c: data[t, z, c],
    )
    monkeypatch.setattr(inference, "_metadata", lambda _: (image, {"id": "source"}))
    settings = {"path": str(source), "channel": 1}
    output = tmp_path / "cache"
    cached, description = inference.prepare_full_counts(settings, output)
    np.testing.assert_array_equal(cached, data[:, :, 1])
    assert description["reverse_z_for_display"] is True
    changed = np.load(output / "counts.npy", mmap_mode="r+")
    changed[0, 0, 0, 0] ^= 1
    changed.flush()
    with pytest.raises(ValueError, match="cache has changed"):
        inference.prepare_full_counts(settings, output)


def test_normalization_checkpoint_is_idempotent_and_detects_changed_probabilities(tmp_path):
    """Finalization can resume after count calibration without applying the scale twice."""
    values = np.lib.format.open_memmap(
        tmp_path / "expected_counts.npy", mode="w+", dtype=np.float32, shape=(2, 3, 2, 2)
    )
    values[:] = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 2, 2)
    chunks = [
        {
            "observed_photons": 123,
            "predicted_mass": float(values.sum()),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
    ]
    first = finalize(tmp_path, values, chunks)
    second = finalize(tmp_path, values, chunks)
    np.testing.assert_allclose(values.sum(dtype=np.float64), 123, rtol=1e-6)
    assert first == second
    q = np.load(tmp_path / "probability.npy", mmap_mode="r+")
    q[0, 0, 0, 0] += 0.01
    q.flush()
    with pytest.raises(ValueError, match="normalization checkpoint"):
        finalize(tmp_path, values, chunks)
