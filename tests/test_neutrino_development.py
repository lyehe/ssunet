"""Verify development-only access and meaningful timing/coupling controls."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.ssunet.neutrino_detection import PhotonWindows
from src.ssunet.neutrino_detection_probe import predict_head
from src.ssunet.neutrino_development import (
    DevelopmentStore,
    diagnostic_head_path,
    pdf_hidden_indices,
    permute_sensor_time_pairings,
    predict_timing,
    ridge_timing_path,
    validation_tail_sensitivity,
)
from src.tools.neutrino_development import _score_parts, _timing_targets, run_diagnostics


@pytest.mark.parametrize(
    "forbidden",
    [
        "test/features.npz",
        "test_results.json",
        "calibration_scores.npz",
        "calibration/observations.npz",
        "../train/features.npz",
    ],
)
def test_store_rejects_test_calibration_and_traversal_before_opening(tmp_path, forbidden):
    """The data boundary applies even to hashing and paths that do not exist."""
    store = DevelopmentStore(tmp_path)
    with pytest.raises(PermissionError, match="Development-only access"):
        store.digest(forbidden)
    assert not store.accessed


def test_store_records_only_allowed_inputs_and_reads_numeric_scalars(tmp_path):
    """A permitted metadata read does not follow any paths named inside it."""
    (tmp_path / "config.json").write_text(json.dumps({"source": "test_results.json"}))
    (tmp_path / "train").mkdir()
    np.savez(tmp_path / "train/features.npz", values=np.arange(5), scalar=np.array(3))
    store = DevelopmentStore(tmp_path)
    assert store.json("config.json")["source"] == "test_results.json"
    data = store.arrays("train/features.npz")
    np.testing.assert_array_equal(data["values"], np.arange(5))
    assert data["scalar"] == 3
    assert store.accessed == {"config.json", "train/features.npz"}


def test_pairing_intervention_preserves_both_exact_marginals_and_empty_windows():
    """Break joint structure while conserving A/B identities and repeated photons."""
    original = PhotonWindows(
        np.array([0, 0, 1, 7]),
        np.array([7, 0, 5, 10, 3, 4, 9]),
        np.array([True, True, True, True, False, False, False]),
        np.array([0.1, 0.2, 0.3]),
    )
    changed, flags = permute_sensor_time_pairings(original, 4, 7)
    a, b = original.dense_batch(0, 3, (3, 4))
    pa, pb = changed.dense_batch(0, 3, (3, 4))
    np.testing.assert_array_equal(a.sum(1), pa.sum(1))
    np.testing.assert_array_equal(a.sum(2), pa.sum(2))
    np.testing.assert_array_equal(b.sum(1), pb.sum(1))
    np.testing.assert_array_equal(b.sum(2), pb.sum(2))
    np.testing.assert_array_equal(original.retained, changed.retained)
    np.testing.assert_array_equal(original.offsets, changed.offsets)
    assert not flags["half_joint_changed"][0]
    assert not flags["half_joint_changed"][1]
    assert flags["eligible"][2]
    assert flags["half_joint_changed"][2]
    np.testing.assert_array_equal(original.locations, [7, 0, 5, 10, 3, 4, 9])


def test_tail_partitions_handle_constant_score_ties_with_shared_randomizers():
    """A constant statistic supplies no ranking information regardless of partition."""
    truth = {
        "expected_signal": np.r_[np.zeros(100), np.full(8, 2)],
        "source_group": np.r_[np.full(100, -1), np.repeat(np.arange(4), 2)],
    }
    randomizers = np.linspace(0.005, 0.995, 108)
    records = validation_tail_sensitivity(
        np.zeros(108), truth, randomizers, seed=12, partitions=5, rates=[0.1, 0.01], strength=2
    )
    for row in records:
        expected = randomizers <= row["target_false_positive_rate"]
        assert row["false_positive_rate"] == expected[:100].mean()
        assert row["primary_efficiency"] == expected[100:].mean()
    assert len(records) == 10


def test_pdf_and_hidden_blocks_partition_the_frozen_feature_vector():
    """Do not mix hidden channels into a PDF-only diagnostic or drop A/B differences."""
    pdf, hidden = pdf_hidden_indices(775, 64, 64, 192)
    assert len(pdf) == 391
    assert len(hidden) == 384
    assert not np.intersect1d(pdf, hidden).size
    np.testing.assert_array_equal(np.sort(np.r_[pdf, hidden]), np.arange(775))
    with pytest.raises(ValueError, match="layout"):
        pdf_hidden_indices(774, 64, 64, 192)


def test_fixed_ridge_recovers_timing_units_and_freezes_training_preprocessing():
    """A simple timing relation is learned without using validation scaling."""
    torch.set_num_threads(1)
    x = np.linspace(-2, 2, 80)
    base = np.column_stack([x, np.ones(80) * 3]).astype(np.float32)
    extra = np.zeros((80, 2), dtype=np.float32)
    targets = np.column_stack([500 + 100 * x, 700 - 50 * x])
    settings = {"feature_scale_floor": 1e-6, "feature_clip": 20}
    head = ridge_timing_path(base, extra, targets, settings, [0.01])[0]
    predictions = predict_timing(head, base, extra, batch_size=7)
    assert np.abs(predictions - targets).mean() < 2
    np.testing.assert_allclose(
        predictions, predict_timing(head, base, extra, batch_size=80), atol=1e-10
    )
    np.testing.assert_allclose(head["mean"][:2], [0, 3], atol=1e-8)
    before = head["mean"].copy()
    outlier = predict_timing(
        head, np.array([[10000, -5000]], dtype=np.float32), np.zeros((1, 2), dtype=np.float32)
    )
    assert np.isfinite(outlier).all()
    np.testing.assert_array_equal(head["mean"], before)


def test_expected_timing_excludes_background_but_keeps_every_positive_label():
    """A zero observed count must not erase a defined positive injection target."""
    labels = {
        "expected_signal": np.array([0, 0.5, 2]),
        "source_group": np.array([-1, 0, 0]),
        "sources_m": np.array([[0, 0, -500]]),
    }
    reference = {
        "positions": np.array([[0, 0, -500], [100, 0, -500]]),
        "edges": np.linspace(0, 6400, 257),
    }
    selected, targets = _timing_targets(labels, reference)
    np.testing.assert_array_equal(selected, [False, True, True])
    np.testing.assert_array_equal(targets[0], targets[1])
    assert targets[0, 1] <= targets[0, 2] <= targets[0, 3]
    assert 0 < targets[0, 0] < 6400


def test_logit_decomposition_matches_the_saved_head_score_with_clipping():
    """Raw and added contributions sum to the original score, including its intercept."""
    base = np.array([[1, 4], [-1, -4]], dtype=np.float32)
    extra = np.array([[2, -2], [-2, 2]], dtype=np.float32)
    head = {
        "mean": np.zeros(4),
        "scale": np.ones(4),
        "weight": np.array([0.1, 0.2, 0.3, 0.4]),
        "intercept": 0.7,
        "clip": 2,
        "extra_dimension": 2,
    }
    raw, added = _score_parts(head, base, extra)
    np.testing.assert_allclose(
        raw + added + head["intercept"], predict_head(head, base, extra), atol=1e-12
    )


def test_diagnosis_cannot_overwrite_a_completed_directory(tmp_path):
    """An existing result is rejected before any source data can be opened."""
    config = yaml.safe_load(Path("config/neutrino_development_diagnostics.yml").read_text())
    config["output"] = str(tmp_path)
    (tmp_path / "results.json").write_text("preserve")
    with pytest.raises(FileExistsError):
        run_diagnostics(config, "cpu")
    assert (tmp_path / "results.json").read_text() == "preserve"


def test_diagnostic_path_keeps_unconverged_candidates_without_recommending_them():
    """A fixed-budget failed grid point remains evidence instead of aborting diagnosis."""
    torch.set_num_threads(1)
    config = yaml.safe_load(Path("config/neutrino_detection_probe.yml").read_text())
    config["head"]["max_iterations"] = 1
    config["head"]["penalties"] = [0.01]
    coordinate = np.linspace(-2, 2, 40).astype(np.float32)
    values = {"base": coordinate[:, None], "self_split": np.zeros((40, 2), dtype=np.float32)}
    labels = (coordinate > 0).astype(float)
    heads, records = diagnostic_head_path(
        values, labels, values, labels, "raw_prior", config, lambda: None
    )
    assert len(heads) == 1
    assert records[0]["iteration_budget_exhausted"]
    assert not records[0]["passes_original_gradient_gate"]
    assert np.isfinite(predict_head(heads[0], values["base"], None)).all()
