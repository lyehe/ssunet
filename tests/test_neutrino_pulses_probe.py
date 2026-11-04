"""Test the real pulse observation contract and the separate labelled probe math."""

import json

import h5py
import numpy as np
import pytest
import torch

from src.ssunet.datasets.neutrino import write_photon_shard
from src.ssunet.datasets.neutrino_pulses import MarkedPulseDataset
from src.ssunet.models.neutrino_pulses import DenseMarkedPulseReference
from src.ssunet.neutrino_pulse_baselines import smooth_pulse_time
from src.ssunet.photon_reference import photon_count_log_probabilities
from src.tools import neutrino_pulse_study
from src.tools.neutrino_probe import (
    angular_errors,
    distribution_features,
    fit_ridge,
    paired_mean_interval,
    predict_ridge,
)
from src.tools.neutrino_pulse_prepare import match_geometry, overlapping_event_groups


def test_charge_marks_preserve_whole_pulses_and_reject_photon_semantics(tmp_path):
    """Four fractional-PE observations remain four split units, regardless of PE sum."""
    path = tmp_path / "pulses.h5"
    write_photon_shard(
        path,
        offsets=np.array([0, 4, 4]),
        sensor_index=np.array([0, 1, 3, 2]),
        time_ns=np.array([10, 15, 25, 35.0]),
        positions_m=np.repeat([[0, 0, 0], [0, 0, 20]], 2, axis=0),
        live=np.ones(4, dtype=bool),
        event_ids=np.array([1, 2]),
        configuration_ids=np.array([1, 2]),
        partition=np.array([0, 0]),
        time_edges_ns=np.linspace(0, 40, 5),
        provenance={
            "kind": "empirical_real_marked_pulses",
            "unit_photon_detections": False,
            "charge_is_photon_count": False,
            "charge_interior_edges_pe": [0.5],
        },
    )
    with h5py.File(path, "a") as handle:
        handle.create_dataset("pulses/charge_pe", data=[0.271, 0.75, 2.75, 0.0])
    dataset = MarkedPulseDataset(path, "train", lazy=True)
    item = dataset[(0, 0)]
    assert int((item["input"] + item["target"]).sum()) == 4
    assert not dataset[1]["input"].any()
    assert not dataset[1]["target"].any()
    dataset.close()
    with h5py.File(path, "a") as handle:
        provenance = json.loads(handle.attrs["provenance_json"])
        provenance["unit_photon_detections"] = True
        handle.attrs["provenance_json"] = json.dumps(provenance)
    with pytest.raises(ValueError, match="non-photon pulse contract"):
        MarkedPulseDataset(path, "train", lazy=True)


def test_marked_dense_model_is_normalized_and_cell_permutation_equivariant():
    """Joint location/time/charge probabilities remain valid after cell reordering."""
    torch.set_num_threads(1)
    model = DenseMarkedPulseReference(marks=2, channels=2)
    counts = torch.arange(64).reshape(1, 4, 16)
    positions = torch.tensor(
        [[0, 0, 0], [0, 0, 0], [10, 20, 30], [10, 20, 30]], dtype=torch.float32
    )
    live, p = torch.ones(4, dtype=torch.bool), torch.tensor([0.5])
    output = model(counts, positions, live, p)
    order = torch.tensor([2, 3, 0, 1])
    reordered = model(counts[:, order], positions[order], live[order], p)
    torch.testing.assert_close(output[:, order], reordered)
    probability = photon_count_log_probabilities(output, live).exp()
    torch.testing.assert_close(probability.sum(), torch.tensor(1.0))
    assert probability.reshape(1, 2, 2, 16).sum(2).shape == (1, 2, 16)


def test_pulse_time_smoothing_preserves_mass_at_clock_edges():
    """Smoothing neither loses edge detections nor mixes independent mark rows."""
    counts = torch.zeros((2, 3, 8), dtype=torch.float64)
    counts[0, 0, 0], counts[1, 2, -1] = 3, 5
    actual = smooth_pulse_time(counts, 1)
    torch.testing.assert_close(actual.sum(-1), counts.sum(-1))
    assert not actual[0, 1:].any()
    assert not actual[1, :2].any()


def test_spatial_cells_do_not_silently_claim_dom_identity():
    """Ambiguous native identities fail, while explicit spatial quantization is auditable."""
    points = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    reference = np.array([[0.0, 0, 0], [20.0, 0, 0]])
    with pytest.raises(ValueError, match="unique bounded"):
        match_geometry(points, reference, 2)
    cells, fixed, distances = match_geometry(points, reference, 2, spatial_cells=True)
    np.testing.assert_array_equal(cells, [0, 0])
    np.testing.assert_array_equal(fixed, reference)
    np.testing.assert_allclose(distances, [0, 1])


def test_overlapping_real_acquisition_records_stay_in_one_family():
    """Grouping uses the stored 0.1 ns start ticks and microsecond durations."""
    groups = overlapping_event_groups(np.array([0, 50, 500]), np.array([0.01, 0.01, 0.01]), 5)
    assert groups[0] == groups[1]
    assert groups[1] != groups[2]


def test_probe_uses_common_feature_dimensions_and_train_only_scaling():
    """A frozen labelled regression generalizes a known linear relation to held-out rows."""
    probability = np.ones((3, 2, 8)) / 16
    features = distribution_features(probability, np.array([0, 1, 5]), coarse_bins=4)
    assert features.shape == (3, 10)
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = (2 * x[:, :1]) + 3
    probe = fit_ridge(x, y, 1e-8)
    np.testing.assert_allclose(predict_ridge(probe, x[-2:]), y[-2:], atol=1e-5)
    np.testing.assert_allclose(probe["feature_mean"], x.mean(0))


def test_probe_angles_and_group_bootstrap_do_not_inflate_variants():
    """Check spherical endpoints and common group resampling across related records."""
    directions = np.array([[0, 0, 1.0], [0, 0, -1.0], [0, 0, 0.0]])
    np.testing.assert_allclose(
        angular_errors(directions, np.tile([0, 0, 1.0], (3, 1))), [0, 180, 0]
    )
    delta, groups = np.array([-1, 1, 2.0]), np.array([1, 1, 2])
    a = paired_mean_interval(delta, groups, draws=251, chunk=7)
    b = paired_mean_interval(delta, groups, draws=251, chunk=64)
    assert a == b


def test_marked_pulse_study_runs_end_to_end_with_frozen_nonphoton_contract(tmp_path, monkeypatch):
    """Exercise selection, identity pinning and first real-style test scoring together."""
    monkeypatch.setattr(neutrino_pulse_study, "_peak_rss", lambda: 1.0)
    config = {
        "data": {
            "output": str(tmp_path),
            "charge_interior_edges_pe": [0.5],
            "split_probability": 0.5,
            "photon_split_seed": 42,
        },
        "model": {"channels": 2},
        "contract": {"count_object": "reconstructed_pulses", "charge_is_photon_count": False},
        "training": {
            "cpu_threads": 1,
            "seeds": [42],
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "max_training_seconds_per_seed": 30,
            "max_evaluation_seconds_per_seed": 30,
            "max_process_rss_gib": 4,
            "max_gpu_allocated_gib": 12,
        },
        "baselines": {
            "prior_pseudocount": 0.0001,
            "time_width_bins": [0, 1],
            "prior_fractions": [0.25, 0.5],
        },
        "evaluation": {
            "bootstrap_seed": 17,
            "bootstrap_draws": 31,
            "confidence_scope": "test_fixture",
        },
    }
    rng = np.random.default_rng(23)
    for label, name in enumerate(("train", "validation", "test")):
        size = 6 if name == "train" else 4
        write_photon_shard(
            tmp_path / f"{name}.h5",
            offsets=np.arange(size + 1) * 16,
            sensor_index=rng.integers(4, size=size * 16),
            time_ns=rng.uniform(0, 40, size=size * 16),
            positions_m=np.repeat([[0, 0, 0], [0, 0, 20]], 2, axis=0),
            live=np.ones(4, dtype=bool),
            event_ids=np.arange(size) + label * 100,
            configuration_ids=np.arange(size) + label * 100,
            partition=np.full(size, label, dtype=np.uint8),
            time_edges_ns=np.linspace(0, 40, 17),
            provenance={
                "kind": "empirical_real_marked_pulses",
                "unit_photon_detections": False,
                "charge_is_photon_count": False,
                "charge_interior_edges_pe": [0.5],
            },
        )
    neutrino_pulse_study.train_pulses(config, "cpu")
    assert (tmp_path / "analysis_lock.json").exists()
    assert not (tmp_path / "test_opened.json").exists()
    neutrino_pulse_study.test_pulses(config, "cpu")
    result = json.loads((tmp_path / "test_results.json").read_text())
    assert result["scores"]["model"]["all"]["held_out_pulses"] > 0
    assert result["no_physical_no_event_inference"]
    with pytest.raises(FileExistsError, match="already has results"):
        neutrino_pulse_study.test_pulses(config, "cpu")
