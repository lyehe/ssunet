"""Check evidence-corruption guards without allocating a model or using the NAS."""

import copy
import hashlib

import pytest

from src.tools.confocal_evaluate import _check_scores, _validate_sources, audit


def test_source_audit_checks_current_and_archived_bytes(tmp_path, monkeypatch):
    """Neither a changed live source nor a changed archive may pass replay validation."""
    monkeypatch.chdir(tmp_path)
    original = b"frozen training code\n"
    current = tmp_path / "model.py"
    current.write_bytes(original)
    study = tmp_path / "study"
    archived = study / "source_snapshot/model.py"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(original)
    expected = {"model.py": hashlib.sha256(original).hexdigest()}
    _validate_sources(study, expected)
    current.write_bytes(b"edited\n")
    with pytest.raises(ValueError, match="Current source"):
        _validate_sources(study, expected)
    current.write_bytes(original)
    archived.write_bytes(b"edited archive\n")
    with pytest.raises(ValueError, match="Archived source"):
        _validate_sources(study, expected)


@pytest.mark.parametrize("bad", ["photons", "nan", "score", "mass"])
def test_checkpoint_score_audit_rejects_corrupt_evidence(bad):
    """Changes in photon draws, nonfinite scores, predictions or mass cannot pass."""
    expected = {
        "photon_nll": 3.0,
        "patch_mean_nll": 3.0,
        "per_patch_total_nll": [10.0, 20.0],
        "per_patch_target_photons": [5, 5],
        "max_probability_mass_error": 0.0,
    }
    actual = copy.deepcopy(expected)
    if bad == "photons":
        actual["per_patch_target_photons"][0] = 6
    elif bad == "nan":
        actual["photon_nll"] = float("nan")
    elif bad == "score":
        actual["photon_nll"] = 4.0
    else:
        actual["max_probability_mass_error"] = 0.1
    message = {
        "photons": "photon draws",
        "nan": "Nonfinite",
        "score": "likelihood",
        "mass": "probability mass",
    }[bad]
    with pytest.raises(ValueError, match=message):
        _check_scores(actual, expected)


def test_gpu_replay_refuses_an_active_training_job_before_loading_torch(tmp_path):
    """An optional audit cannot compete with the live 4D worker for GPU memory."""
    path = tmp_path / "training/pilot/process.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"status":"running"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="finish"):
        audit(tmp_path)
