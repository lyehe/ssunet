"""Check the paper-aligned models and deterministic resumable method training."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


@pytest.mark.parametrize("architecture", ["dense_reference", "full_time_reference", "conv4d"])
def test_gap_encoding_keeps_sensor_order_and_physical_probability_support(architecture):
    """All arms use the same raw-count encoding and normalize only physical support."""
    pytest.importorskip("torch_convnd")
    from src.ssunet.photon_reference import photon_count_log_probabilities
    from src.tools.neutrino_gap4d_retry import make_model

    torch.set_num_threads(1)
    positions = np.asarray([[x, 0, z] for x in (0, 10) for z in (0, 10)])
    settings = {"name": architecture, "channels": 2, "encoding_levels": 10, "backend": "torch"}
    model = make_model(settings, positions)
    counts = torch.arange(32).reshape(1, 4, 8) % 7
    live = torch.tensor([True, True, True, False])
    arguments = torch.tensor(positions).float(), live, torch.tensor([0.5])
    logits = model(counts, *arguments)
    changed = counts.clone()
    changed[:, -1] = 10000
    torch.testing.assert_close(logits, model(changed, *arguments))
    q = photon_count_log_probabilities(logits, live).exp()
    torch.testing.assert_close(q.sum((1, 2)), torch.ones(1))
    assert torch.count_nonzero(q[:, -1]) == 0
    order = torch.tensor([3, 1, 0, 2])
    permuted = model(counts[:, order], arguments[0][order], live[order], arguments[2])
    torch.testing.assert_close(permuted, logits[:, order])


def test_gap_training_resumes_without_changing_optimizer_trajectory(tmp_path, monkeypatch):
    """An interrupted mid-epoch GAP run reproduces a continuous fixed-budget run."""
    pytest.importorskip("torch_convnd")
    from src.ssunet.datasets.neutrino import partition_configurations, write_photon_shard
    from src.tools import neutrino_gap4d_retry as study

    torch.set_num_threads(1)
    positions = np.asarray([[x, 0, z] for x in (0, 10) for z in (0, 10)])
    identifiers = np.arange(80, dtype=np.int64)
    photons = np.where(identifiers % 5 == 0, 0, 8)
    path = tmp_path / "counts.h5"
    write_photon_shard(
        path,
        offsets=np.concatenate(([0], np.cumsum(photons))).astype(np.int64),
        sensor_index=np.repeat(identifiers % 4, photons).astype(np.int32),
        time_ns=np.full(int(photons.sum()), 10.0),
        positions_m=positions,
        live=np.ones(4, dtype=bool),
        event_ids=identifiers,
        configuration_ids=identifiers,
        partition=partition_configurations(identifiers, seed=7),
        time_edges_ns=np.arange(9) * 25.0,
        provenance={"kind": "gap_resume_test"},
    )
    config = yaml.safe_load(Path("config/neutrino_gap4d_retry.yml").read_text())
    config.update(dataset=str(path), dataset_sha256=study.reference.shard_sha256(path))
    config["training"].update(steps=4, seeds=[42], batch_size=8)
    config["evaluation"].update(steps=[0, 2, 4], donor_draws=1, fixed_training_events=8)
    data, train, valid, diagnostic, choices = study.development_views(config)
    settings = {"name": "conv4d", "channels": 2, "encoding_levels": 10, "backend": "torch"}
    device = torch.device("cpu")
    study.train_seed(
        config,
        settings,
        42,
        data,
        valid,
        diagnostic,
        (choices, train),
        tmp_path / "complete",
        device,
    )
    original = study.training_step
    calls = 0

    def interrupt(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            message = "interrupted GAP worker"
            raise InterruptedError(message)
        return original(*args)

    monkeypatch.setattr(study, "training_step", interrupt)
    with pytest.raises(InterruptedError, match="GAP worker"):
        study.train_seed(
            config,
            settings,
            42,
            data,
            valid,
            diagnostic,
            (choices, train),
            tmp_path / "partial",
            device,
        )
    monkeypatch.setattr(study, "training_step", original)
    study.train_seed(
        config,
        settings,
        42,
        data,
        valid,
        diagnostic,
        (choices, train),
        tmp_path / "partial",
        device,
    )
    expected = torch.load(tmp_path / "complete/conv4d/seed_42/last.pt", weights_only=True)
    actual = torch.load(tmp_path / "partial/conv4d/seed_42/last.pt", weights_only=True)
    for name, value in expected["state_dict"].items():
        torch.testing.assert_close(actual["state_dict"][name], value, rtol=0, atol=0)
    assert actual["state"]["step"] == 4
    data.close()


def test_method_override_cannot_apply_bit2bit_to_count_data():
    """Enforce the requested acquisition-to-method mapping before training starts."""
    pytest.importorskip("torch_convnd")
    from src.tools.neutrino_gap4d_retry import validate_method

    with pytest.raises(ValueError, match="acquisition"):
        validate_method({"observation_kind": "photon_counts", "method": "bit2bit"})
