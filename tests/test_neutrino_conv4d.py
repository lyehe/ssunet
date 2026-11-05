"""Check physical-grid mapping and the observation-only 4D retry contract."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.ssunet.datasets.neutrino import partition_configurations, shard_sha256, write_photon_shard
from src.ssunet.photon_reference import photon_count_log_probabilities, photon_count_nll


def detector_positions():
    """Use three strings with missing cells in their enclosing Cartesian grid."""
    return np.asarray([[x, y, z] for x, y in ((0, 0), (10, 10), (20, 0)) for z in (0, 10)])


@pytest.fixture
def model_class():
    """Keep this optional experiment out of the core package's dependency contract."""
    pytest.importorskip("torch_convnd")
    from src.ssunet.models.neutrino_conv4d import GridPhotonReference4d

    return GridPhotonReference4d


def test_grid_normalization_inactive_inputs_and_empty_target_gradients(model_class):
    """Missing cells/dead sensors cannot acquire probability or leak input counts."""
    torch.manual_seed(2)
    positions = torch.tensor(detector_positions(), dtype=torch.float32)
    model = model_class(positions.numpy(), channels=2, backend="torch")
    assert model.grid_shape == (3, 2, 2)
    counts = torch.ones(2, 6, 8, requires_grad=True)
    live = torch.tensor([True, True, True, True, True, False])
    arguments = (positions, live, torch.full((2,), 0.5))
    logits = model(counts, *arguments)
    assert logits.shape == counts.shape
    probabilities = photon_count_log_probabilities(logits, live).exp()
    torch.testing.assert_close(probabilities.sum((1, 2)), torch.ones(2))
    assert torch.count_nonzero(probabilities[:, -1]) == 0
    changed = counts.detach().clone()
    changed[:, -1] = 1000000
    torch.testing.assert_close(model(changed, *arguments), logits)
    photon_count_nll(logits, torch.zeros_like(counts, dtype=torch.int64), live).backward()
    assert counts.grad.abs().sum() == 0
    assert all(parameter.grad.abs().sum() == 0 for parameter in model.parameters())


def test_physical_sensor_permutation_preserves_predictions(model_class):
    """Sensor array order must not become a false spatial neighborhood."""
    torch.manual_seed(3)
    positions = torch.tensor(detector_positions(), dtype=torch.float32)
    model = model_class(positions.numpy(), channels=2, backend="torch")
    counts = torch.poisson(torch.full((2, 6, 8), 0.2))
    live, p = torch.ones(6, dtype=torch.bool), torch.full((2,), 0.5)
    order = torch.tensor([4, 2, 0, 5, 1, 3])
    original = model(counts, positions, live, p)
    reordered = model(counts[:, order], positions[order], live[order], p)
    torch.testing.assert_close(reordered, original[:, order])


def test_full_four_axis_kernels_mix_neighbors_without_pooled_metadata(model_class):
    """An input impulse reaches another physical sensor/time through 4D kernels."""
    positions = torch.tensor(detector_positions(), dtype=torch.float32)
    model = model_class(positions.numpy(), channels=2, backend="torch")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.stem.weight[:, 0] = 1
        model.head.weight.fill_(1)
        for block in model.blocks:
            assert block.weight.shape[-4:] == (3, 3, 3, 3)
            block.weight.fill_(0.1)
    counts = torch.zeros(1, 6, 8)
    live, p = torch.ones(6, dtype=torch.bool), torch.full((1,), 0.5)
    before = model(counts, positions, live, p)
    counts[0, 0, 2] = 1
    after = model(counts, positions, live, p)
    assert after[0, 3, 3] > before[0, 3, 3]


def test_grid_rejects_distorted_or_duplicate_sensor_geometry(model_class):
    """Do not silently quantize irregular physical distances into adjacent indices."""
    with pytest.raises(ValueError, match="regularly spaced"):
        model_class(np.asarray([[0, 0, 0], [10, 0, 0], [35, 0, 0]]))
    with pytest.raises(ValueError, match="unique"):
        model_class(np.zeros((2, 3)))


@pytest.mark.usefixtures("model_class")
def test_retry_uses_only_development_counts_and_saves_reloadable_models(tmp_path, monkeypatch):
    """Exercise matched training, independent targets, frozen scores and output protection."""
    settings = yaml.safe_load(Path("config/neutrino_conv4d_retry.yml").read_text(encoding="utf-8"))
    config = copy.deepcopy(settings)
    path = tmp_path / "counts.h5"
    event_ids = np.arange(80, dtype=np.int64)
    count = np.where(event_ids % 5 == 0, 0, 8)
    write_photon_shard(
        path,
        offsets=np.concatenate(([0], np.cumsum(count))).astype(np.int64),
        sensor_index=np.repeat(event_ids % 6, count).astype(np.int32),
        time_ns=np.full(int(count.sum()), 10.0),
        positions_m=detector_positions(),
        live=np.ones(6, dtype=bool),
        event_ids=event_ids,
        configuration_ids=event_ids,
        partition=partition_configurations(event_ids, seed=7),
        time_edges_ns=np.arange(9) * 25.0,
        provenance={"kind": "test_fixture"},
    )
    config.update(
        dataset=str(path), dataset_sha256=shard_sha256(path), output=str(tmp_path / "run")
    )
    config["training"].update(seeds=[42], steps=2, batch_size=8, cpu_threads=1)
    config["evaluation"].update(
        steps=[0, 1, 2], fixed_training_events=8, donor_draws=1, bootstrap_draws=32
    )
    for model in config["models"]:
        model.update(channels=2, backend="torch")
    from src.tools import neutrino_conv4d_retry as runner

    original = runner.NeutrinoPhotonDataset

    def development_only(*args, **kwargs):
        assert args[1] in ("train", "validation")
        assert kwargs["lazy"] is True
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "NeutrinoPhotonDataset", development_only)
    report = runner.run_retry(config, "cpu", render=False)
    assert report["test_or_truth_used"] is False
    assert all(run["completed_fixed_schedule"] for run in report["runs"])
    assert {run["model"] for run in report["runs"]} == {
        "dense_reference",
        "full_time_reference",
        "conv4d",
    }
    checkpoint = torch.load(tmp_path / "run/conv4d/seed_42/last.pt", weights_only=True)
    model = runner.make_model(checkpoint["model"], detector_positions())
    model.load_state_dict(checkpoint["state_dict"])
    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_retry(config, "cpu", render=False)
