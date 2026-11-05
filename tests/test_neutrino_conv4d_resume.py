"""Verify that recovery retains the original photon batches and optimizer trajectory."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


def test_interrupted_training_reproduces_uninterrupted_weights(tmp_path, monkeypatch):
    """Restoring mid-epoch must reproduce the next data/split and AdamW states."""
    pytest.importorskip("torch_convnd")
    from src.tools import neutrino_conv4d_resume as recovery
    from src.tools import neutrino_conv4d_retry as runner

    torch.set_num_threads(1)
    path = tmp_path / "photons.h5"
    identifiers = np.arange(80, dtype=np.int64)
    photons = np.where(identifiers % 5 == 0, 0, 8)
    positions = np.asarray([[x, y, z] for x in (0, 10) for y in (0, 10) for z in (0, 10)])
    from src.ssunet.datasets.neutrino import partition_configurations, write_photon_shard

    write_photon_shard(
        path,
        offsets=np.concatenate(([0], np.cumsum(photons))).astype(np.int64),
        sensor_index=np.repeat(identifiers % 8, photons).astype(np.int32),
        time_ns=np.full(int(photons.sum()), 10.0),
        positions_m=positions,
        live=np.ones(8, dtype=bool),
        event_ids=identifiers,
        configuration_ids=identifiers,
        partition=partition_configurations(identifiers, seed=7),
        time_edges_ns=np.arange(9) * 25.0,
        provenance={"kind": "recovery_test"},
    )
    config = yaml.safe_load(Path("config/neutrino_conv4d_retry.yml").read_text())
    config.update(dataset=str(path), dataset_sha256=runner.shard_sha256(path))
    config["training"].update(steps=4, batch_size=8, seeds=[42])
    config["evaluation"].update(steps=[0, 1, 2, 4])
    settings = {"name": "conv4d", "channels": 2, "backend": "torch"}
    training = runner.NeutrinoPhotonDataset(path, "train", lazy=True)
    frozen = runner.NeutrinoPhotonDataset(path, "train", training=False, lazy=True)
    validating = runner.NeutrinoPhotonDataset(path, "validation", lazy=True)
    train_view, validation = (
        runner.freeze_photon_view(frozen),
        runner.freeze_photon_view(validating),
    )
    diagnostic = runner.subset_view(train_view, np.arange(8))
    choices = runner.matched_donor_indices(
        train_view.inputs.sum((1, 2)).numpy(),
        validation.inputs.sum((1, 2)).numpy(),
        seed=914,
        draws=1,
    )
    donors = choices, train_view
    device = torch.device("cpu")
    runner.train_arm(
        config,
        settings,
        42,
        training,
        validation,
        diagnostic,
        donors,
        device,
        tmp_path / "complete",
    )
    update = runner.training_step
    calls = 0

    def interrupt(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            message = "simulated worker interruption"
            raise InterruptedError(message)
        return update(*args)

    monkeypatch.setattr(runner, "training_step", interrupt)
    with pytest.raises(InterruptedError, match="simulated"):
        runner.train_arm(
            config,
            settings,
            42,
            training,
            validation,
            diagnostic,
            donors,
            device,
            tmp_path / "partial",
        )
    monkeypatch.setattr(runner, "training_step", update)
    recovered, values, _ = recovery.continue_arm(
        config,
        settings,
        42,
        training,
        validation,
        diagnostic,
        donors,
        device,
        tmp_path / "partial",
        previous_seconds=0.0,
    )
    assert recovered["recovery"]["from_step"] == 2
    original = torch.load(tmp_path / "complete/conv4d/seed_42/last.pt", weights_only=True)
    actual = torch.load(tmp_path / "partial/conv4d/seed_42/last.pt", weights_only=True)
    for key, expected in original["state_dict"].items():
        torch.testing.assert_close(actual["state_dict"][key], expected, rtol=0, atol=0)
    with np.load(tmp_path / "complete/conv4d/seed_42/scores.npz") as stored:
        np.testing.assert_array_equal(values, stored["model"])
    for dataset in (training, frozen, validating):
        dataset.close()


def test_recovery_timing_requires_the_matching_saved_checkpoint(tmp_path):
    """PowerShell UTF-16 logs must not supply timing from another optimizer step."""
    pytest.importorskip("torch_convnd")
    from src.tools.neutrino_conv4d_resume import logged_training_seconds

    path = tmp_path / "run.log"
    path.write_text("conv4d seed=43 step=1000 all=2.5 low=3.5 training_s=30.0\n", encoding="utf-16")
    assert logged_training_seconds(path, "conv4d", 43, 1000) == 30.0
    with pytest.raises(ValueError, match="No timing"):
        logged_training_seconds(path, "conv4d", 43, 500)
