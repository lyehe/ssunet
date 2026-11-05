"""Verify temporal separation, photon conservation and shared 3D/4D supervision."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.ssunet.datasets import confocal
from src.ssunet.datasets.confocal import sample_split
from src.ssunet.models.confocal_gap import ConfocalGAPUNet, confocal_gap_loss
from src.ssunet.photon_methods import paper_photon_loss
from src.tools.confocal_train import evaluate, load_config


@pytest.mark.parametrize("dimensions", [3, 4])
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_confocal_model_preserves_odd_spatial_shape_and_backpropagates(dimensions, depth):
    """Pad only inside the network and return gradients at every real output voxel."""
    if dimensions == 4:
        pytest.importorskip("torch_convnd")
    torch.set_num_threads(1)
    model = ConfocalGAPUNet(dimensions=dimensions, base_channels=2, depth=depth, backend="torch")
    inputs = torch.arange(2 * 5 * 7 * 9).reshape(1, 1, 2, 5, 7, 9) % 4
    target = torch.ones_like(inputs)
    logits = model(inputs)
    assert logits.shape == inputs.shape
    confocal_gap_loss(logits, target, inputs).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_conv3d_control_never_mixes_time_points():
    """Changing another time point cannot change this control's volume logits."""
    torch.set_num_threads(1)
    torch.manual_seed(32)
    model = ConfocalGAPUNet(dimensions=3, base_channels=2)
    inputs = torch.ones(2, 1, 3, 4, 4, 4, dtype=torch.int64)
    changed = inputs.clone()
    changed[:, :, 1] = 3
    with torch.no_grad():
        original, perturbed = model(inputs), model(changed)
    torch.testing.assert_close(original[:, :, [0, 2]], perturbed[:, :, [0, 2]], rtol=0, atol=0)


def test_conv4d_uses_neighbouring_time_points():
    """The full kernel can use temporal information absent from the 3D control."""
    pytest.importorskip("torch_convnd")
    torch.set_num_threads(1)
    torch.manual_seed(32)
    model = ConfocalGAPUNet(dimensions=4, base_channels=2, backend="torch")
    inputs = torch.ones(1, 1, 3, 4, 4, 4, dtype=torch.int64)
    changed = inputs.clone()
    changed[:, :, 1] = 3
    with torch.no_grad():
        difference = (model(inputs)[:, :, 0] - model(changed)[:, :, 0]).abs().max()
    assert difference > 1e-6


def test_confocal_objective_matches_existing_gap_value_and_gradient():
    """Both model modes share the verified whole-patch GAP target normalization."""
    torch.manual_seed(8)
    logits = torch.randn(2, 1, 2, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    target = torch.arange(logits.numel()).reshape(logits.shape) % 3
    inputs = torch.ones_like(target)
    actual = confocal_gap_loss(logits, target, inputs)
    expected = paper_photon_loss(
        logits.reshape(2, 1, -1),
        target.reshape(2, 1, -1),
        torch.ones(2, 1, dtype=torch.bool),
        inputs.reshape(2, 1, -1),
        observation_kind="photon_counts",
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_patch_draws_conserve_photons_and_ignore_global_model_rng():
    """Draw equal A/B patches for both architectures without normalizing counts."""
    counts = np.arange(4 * 4 * 8 * 8, dtype=np.uint16).reshape(4, 4, 8, 8) % 7
    options = {"batch_size": 2, "p": 0.5, "seed": 42, "index": 2, "stream": 0}
    a, b = sample_split(counts, [4, 4, 8, 8], **options)
    torch.manual_seed(101)
    np.random.seed(9)
    again_a, again_b = sample_split(counts, [4, 4, 8, 8], **options)
    assert a.dtype == b.dtype == torch.int64
    torch.testing.assert_close(
        a + b, torch.from_numpy(counts.astype(np.int64))[None, None].expand_as(a)
    )
    torch.testing.assert_close(a, again_a, rtol=0, atol=0)
    torch.testing.assert_close(b, again_b, rtol=0, atol=0)
    fresh_a, _ = sample_split(counts, [4, 4, 8, 8], **{**options, "index": 3})
    assert not torch.equal(a, fresh_a)


@pytest.mark.parametrize("bad", [np.ones((4, 4, 8, 8)), -np.ones((4, 4, 8, 8), dtype=int)])
def test_patch_sampler_rejects_normalized_or_negative_counts(bad):
    """Float intensity arrays cannot silently masquerade as splittable photon counts."""
    with pytest.raises(ValueError, match="nonnegative"):
        sample_split(bad, [2, 2, 4, 4], batch_size=1, p=0.5, seed=42, index=0, stream=0)


def test_uniform_prediction_has_exact_expected_held_out_photon_score():
    """Validate distribution scoring without a clean image or a noisy-image PSNR."""

    class Uniform(torch.nn.Module):
        def forward(self, inputs):
            return torch.zeros_like(inputs, dtype=torch.float32)

    inputs = torch.zeros(1, 1, 2, 2, 4, 4, dtype=torch.int64)
    target = torch.zeros_like(inputs)
    target[..., 0, 0, 0, 0] = 3
    result = evaluate(Uniform(), [(inputs, target)], torch.device("cpu"))
    assert result["target_photons"] == 3
    assert result["photon_nll"] == pytest.approx(math.log(64), abs=1e-6)
    assert result["max_probability_mass_error"] < 1e-6


@pytest.mark.parametrize(
    ("path", "expected_range"),
    [
        ("config/confocal_gap4d_test.yml", (500_000, 1_000_000)),
        ("config/confocal_gap4d_large.yml", (9_000_000, 11_000_000)),
    ],
)
def test_default_models_have_similar_parameter_budgets(path, expected_range):
    """Keep the 3D control within five percent of the 4D pilot's parameter count."""
    pytest.importorskip("torch_convnd")
    config = load_config(path)
    sizes = [
        sum(
            p.numel()
            for p in ConfocalGAPUNet(
                **{k: v for k, v in settings.items() if k not in {"name", "backend"}},
                backend="torch",
            ).parameters()
        )
        for settings in config["models"]
    ]
    assert expected_range[0] < sizes[0] < expected_range[1]
    assert abs(sizes[0] - sizes[1]) / sizes[0] < 0.05


@pytest.mark.parametrize("depth", [0, 1, 2.5, True])
def test_model_rejects_invalid_depth_before_allocating_layers(depth):
    """Depth must denote an integer count of encoder resolutions, at least two."""
    with pytest.raises(ValueError, match="depth"):
        ConfocalGAPUNet(depth=depth)


def test_cache_preserves_selected_count_planes_and_detects_modified_pixels(tmp_path, monkeypatch):
    """Exercise crop coordinates, channel selection, cache reuse and corruption detection."""

    class Source:
        dims = SimpleNamespace(t=4, z=4, y=8, x=8)
        channels = 2
        calls = 0

        def get_frame(self, *, t, z, c):
            self.calls += 1
            y, x = np.indices((8, 8))
            return (20 * t + 5 * z + 2 * c + y + x).astype(np.uint8)

    image = Source()
    source = {"channel": 1, "crop_tzyx": [[1, 4], [1, 3], [2, 6], [3, 7]]}
    identity = {"source": source, "conversion": "identity"}
    monkeypatch.setattr(confocal, "_metadata", lambda _: (image, identity))
    destination = tmp_path / "counts.npy"
    path, manifest = confocal.prepare_counts(source, destination)
    counts = np.load(path, allow_pickle=False)
    expected = np.stack(
        [np.stack([image.get_frame(t=t, z=z, c=1)[2:6, 3:7] for z in [1, 2]]) for t in [1, 2, 3]]
    )
    np.testing.assert_array_equal(counts, expected)
    assert counts.dtype == np.uint8
    assert sum(manifest["histogram"]) == counts.size
    calls = image.calls
    confocal.prepare_counts(source, destination)
    assert image.calls == calls
    counts[0, 0, 0, 0] += 1
    np.save(destination, counts, allow_pickle=False)
    with pytest.raises(ValueError, match="changed"):
        confocal.prepare_counts(source, destination)
