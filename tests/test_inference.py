"""Distribution scaling and temporal-coordinate regression checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from infer_spad_offsets import infer_zwindow
from src.ssunet.inference import (
    average_temporal_offsets,
    logits_to_expected_counts,
    reflected_window,
)
from src.tools import gpuinference
from src.tools.cpuinference import cpu_inference


class IdentityLogWeights(torch.nn.Module):
    """Emit logits for the input count distribution with an arbitrary offset."""

    def __init__(self, shift=0.0):
        super().__init__()
        self.shift = shift

    def forward(self, data):
        """Represent zero photon weights by negligible positive mass."""
        return data.clamp_min(1e-30).log() + self.shift


@pytest.mark.parametrize("length", [1, 7, 32, 33, 64, 65])
def test_offset_predictions_keep_recorded_frame_coordinates(length):
    """Every offset preserves impulses at both boundaries and interior time."""
    data = np.zeros((length, 2, 2, 3), dtype=np.float32)
    for index in {0, length // 2, length - 1}:
        data[index] = index + 1
    saved = {}
    averaged, counts = average_temporal_offsets(
        data,
        lambda window: window,
        window_size=32,
        on_offset=lambda offset, volume: saved.update({offset: volume.copy()}),
    )
    np.testing.assert_array_equal(averaged, data)
    np.testing.assert_array_equal(counts, np.full(length, 16))
    for volume in saved.values():
        np.testing.assert_array_equal(volume, data)
    assert averaged.sum() == data.sum()


def test_reflected_window_matches_numpy_padding():
    """Only temporal context is reflected; recorded coordinate values survive."""
    data = np.arange(7).reshape(7, 1)
    expected = np.pad(data, ((9, 12), (0, 0)), mode="reflect")
    np.testing.assert_array_equal(reflected_window(data, -9, 28), expected)


def test_count_conversion_is_invariant_to_logit_offset():
    """Common logit offsets do not change the predicted count distribution."""
    observed = torch.tensor([1.0, 2.0, 0.0, 5.0]).reshape(1, 1, 1, 2, 2)
    logits = observed.clamp_min(1e-30).log()
    first = logits_to_expected_counts(logits, observed)
    shifted = logits_to_expected_counts(logits + 1000, observed)
    torch.testing.assert_close(first, observed)
    torch.testing.assert_close(shifted, first, atol=1e-4, rtol=1e-4)
    assert shifted.sum().item() == pytest.approx(observed.sum().item())
    empty = logits_to_expected_counts(torch.full_like(logits, 1000), observed * 0)
    torch.testing.assert_close(empty, observed * 0)


def test_2d_count_conversion_normalizes_each_recorded_frame():
    """Independent logit offsets cannot change a frame's observed count."""
    observed = torch.arange(1, 25, dtype=torch.float32).reshape(1, 2, 3, 2, 2)
    shifts = torch.tensor([-1000.0, 0.0, 1000.0]).reshape(1, 1, 3, 1, 1)
    result = logits_to_expected_counts(observed.log() + shifts, observed, dim=(-1, -2, -4))
    torch.testing.assert_close(result, observed, atol=0.001, rtol=1e-4)
    torch.testing.assert_close(result.sum(dim=(-1, -2, -4)), observed.sum(dim=(-1, -2, -4)))


@pytest.mark.parametrize("method", ["cpu", "patch", "offset"])
def test_2d_inference_ignores_arbitrary_per_frame_logit_offsets(method):
    """Both core and SPAD paths respect the photon_2d training axes."""

    class FrameLogWeights(IdentityLogWeights):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(loss_function="photon_2d")

        def forward(self, data):
            shifts = torch.arange(data.shape[2], device=data.device).reshape(1, 1, -1, 1, 1)
            return super().forward(data) + shifts * 200

    data = np.arange(1, 57, dtype=np.float32).reshape(7, 2, 2, 2)
    model = FrameLogWeights()
    if method == "cpu":
        result = cpu_inference(model, data)
    elif method == "patch":
        result = gpuinference.gpu_patch_inference(
            model, data, min_overlap=1, device="cpu", initial_patch_depth=4
        )
    else:
        result = infer_zwindow(model, data.transpose(0, 2, 3, 1), torch.device("cpu")).transpose(
            0, 3, 1, 2
        )
    np.testing.assert_allclose(result, data, atol=0.005, rtol=1e-4)
    np.testing.assert_allclose(result.sum(axis=(1, 2, 3)), data.sum(axis=(1, 2, 3)), rtol=1e-5)


def test_spad_window_preserves_photon_mass_and_time():
    """SPAD window inference supplies the same count units for every offset."""
    data = np.zeros((32, 2, 2, 3), dtype=np.float32)
    data[0, 0, 0, 0] = 1
    data[17, 1, 1, 2] = 1
    data[-1, 0, 1, 1] = 1
    prediction = infer_zwindow(IdentityLogWeights(1000), data, torch.device("cpu"))
    np.testing.assert_allclose(prediction, data, atol=1e-5, rtol=1e-5)
    assert prediction.sum() == pytest.approx(data.sum())


@pytest.mark.parametrize("length", [1, 3, 4, 5, 13])
def test_patch_inference_covers_short_and_partial_sequences(length):
    """Patch overlap cannot introduce zero frames or temporal shifts."""
    data = np.arange(1, length * 2 * 2 + 1, dtype=np.float32).reshape(length, 2, 2)
    result = gpuinference.gpu_patch_inference(
        IdentityLogWeights(1000), data, min_overlap=1, device="cpu", initial_patch_depth=4
    )
    np.testing.assert_allclose(result, data, atol=0.002, rtol=1e-4)


def test_cpu_multichannel_inference_preserves_shape_and_count():
    """Full-volume CPU inference accepts the same ZCXY layout as patch inference."""
    data = np.arange(1, 73, dtype=np.float32).reshape(3, 2, 3, 4)
    prediction = cpu_inference(IdentityLogWeights(1000), data)
    np.testing.assert_allclose(prediction, data, atol=0.003, rtol=1e-4)
    assert prediction.sum() == pytest.approx(data.sum(), rel=1e-5)


def test_short_context_respects_model_temporal_downsampling():
    """Padding supplied to a model is excluded from output count normalization."""

    class RequiresFourFrames(IdentityLogWeights):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(depth=3, z_conv_stage=3)

        def forward(self, data):
            assert data.shape[2] % 4 == 0
            return super().forward(data)

    data = np.full((1, 2, 2), 3, dtype=np.float32)
    prediction = cpu_inference(RequiresFourFrames(), data)
    np.testing.assert_allclose(prediction, data)


def test_grid_inference_averages_matching_spatial_coordinates():
    """Overlapping spatial tiles preserve identity predictions and finite zeros."""
    data = np.arange(1, 5 * 9 * 9 + 1, dtype=np.float32).reshape(5, 9, 9)
    prediction = gpuinference.grid_inference(
        data,
        IdentityLogWeights(),
        torch.device("cpu"),
        split=2,
        patch_size=6,
        min_overlap=1,
        initial_patch_depth=4,
    )
    np.testing.assert_allclose(prediction, data, atol=0.001, rtol=1e-5)


def test_patch_sizer_changes_the_tested_depth(monkeypatch):
    """An OOM retry tests a smaller CZYX tensor instead of repeating the first one."""
    shapes = []

    def probe(model, shape, mixed_precision):
        shapes.append(shape)
        if shape[1] > 16:
            raise torch.cuda.OutOfMemoryError()

    monkeypatch.setattr(gpuinference, "test_model_vram", probe)
    depth = gpuinference.patch_sizer(IdentityLogWeights(), (3, 32, 8, 8), 32, 4)
    assert depth == 16
    assert shapes == [(3, 32, 8, 8), (3, 16, 8, 8)]


@pytest.mark.parametrize("overlap", [-1, 4, 5])
def test_invalid_temporal_overlap_is_rejected(overlap):
    """Invalid overlap produces a direct error instead of broken patch arithmetic."""
    with pytest.raises(ValueError, match="min_overlap"):
        gpuinference.gpu_patch_inference(
            IdentityLogWeights(), np.ones((4, 2, 2)), overlap, "cpu", initial_patch_depth=4
        )


def test_empty_volume_is_rejected():
    """No inference path silently exports an empty or uncovered reconstruction."""
    with pytest.raises(ValueError, match="nonempty"):
        cpu_inference(IdentityLogWeights(), np.empty((0, 2, 2)))
    with pytest.raises(ValueError, match="nonempty"):
        average_temporal_offsets(np.empty((0, 2, 2, 3)), lambda data: data)
