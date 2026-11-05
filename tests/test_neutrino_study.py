"""Independent analysis-lock and resource/conditioning gate checks."""

import json

import numpy as np
import pytest
import torch

from src.ssunet.datasets.neutrino import shard_sha256
from src.ssunet.neutrino_evaluation import PhotonScoreRecords
from src.ssunet.neutrino_training import _new_state, _resource_stop
from src.tools.neutrino_study import (
    _config_hash,
    _score_bundle,
    lock_analysis,
    verify_analysis_lock,
)


def test_training_memory_gate_is_checked_independently_of_optimizer(monkeypatch):
    """A measured memory excess stops work before a training update."""
    monkeypatch.setattr("src.ssunet.neutrino_training._peak_rss", lambda: 5.0)
    config = {
        "training": {
            "max_process_rss_gib": 4,
            "max_gpu_allocated_gib": 12,
            "max_training_seconds_per_seed": 60,
            "max_evaluation_seconds_per_seed": 60,
        }
    }
    state = _new_state()
    assert _resource_stop(state, None, config, torch.device("cpu")) == "memory_budget"
    assert state["peak_process_tree_rss_gib"] == 5.0


def test_brightness_confounded_donor_contrasts_are_not_interpreted():
    """A control is compared only where retained photon totals are exactly matched."""
    records = PhotonScoreRecords(
        np.arange(3), np.arange(3), np.asarray([1, 5, 100]), np.asarray([2, 3, 50])
    )
    values = {
        "model": np.asarray([4.0, 6.0, 100.0]),
        "baseline": np.asarray([6.0, 9.0, 150.0]),
        "matched_donor": np.asarray([8.0, 12.0, 500.0]),
    }
    config = {
        "evaluation": {"bootstrap_seed": 7, "bootstrap_draws": 100, "bootstrap_chunk_draws": 7}
    }
    result = _score_bundle(records, values, config, "baseline", np.asarray([True, True, False]))
    assert result["contrasts"]["all"]["matched_donor"] is None
    assert result["contrasts"]["retained_1_to_10"]["matched_donor"] is not None


def test_failed_development_gate_does_not_open_test(tmp_path):
    """No test file or opening record is needed to reject an ineligible lock."""
    (tmp_path / "development_summary.json").write_text(
        json.dumps({"eligible_for_test_lock": False})
    )
    with pytest.raises(RuntimeError, match="gate"):
        lock_analysis({"data": {"output": str(tmp_path)}})
    assert not (tmp_path / "analysis_lock.json").exists()
    assert not (tmp_path / "test_opened.json").exists()


def test_lock_verification_rejects_changed_training_reference(tmp_path):
    """Frozen baseline/donor references are protected together with model/data/code."""
    config = {"data": {"output": str(tmp_path)}}
    fields = {
        "cohort_manifest_sha256": "cohort_manifest.parquet",
        "baseline_selection_sha256": "baseline_selection.json",
        "training_reference_sha256": "training_reference.npz",
        "training_reference_metadata_sha256": "training_reference.json",
        "development_summary_sha256": "development_summary.json",
    }
    lock = {
        "config_sha256": _config_hash(config),
        "protocol_sha256": shard_sha256("plans/neutrino_study_protocol.txt"),
        "code_sha256": {},
        "checkpoints": {},
        "observations": {},
    }
    for key, name in fields.items():
        path = tmp_path / name
        path.write_bytes(b"fixed evidence")
        lock[key] = shard_sha256(path)
    (tmp_path / "analysis_lock.json").write_text(json.dumps(lock))
    assert verify_analysis_lock(config) == lock
    (tmp_path / "training_reference.npz").write_bytes(b"changed prior")
    with pytest.raises(ValueError, match="changed"):
        verify_analysis_lock(config)
