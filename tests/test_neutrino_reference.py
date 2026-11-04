"""Statistical, boundary, support and leakage checks for photon reference data."""

import json
import sqlite3

import h5py
import numpy as np
import pytest
import torch

from src.ssunet.datasets.neutrino import (
    NeutrinoPhotonDataset,
    bin_photons,
    partition_configurations,
    split_photon_counts,
    write_photon_shard,
)
from src.ssunet.models.neutrino_dense import DensePhotonReference
from src.ssunet.photon_reference import (
    audit_poisson_thinning,
    photon_count_log_probabilities,
    photon_count_nll,
)
from src.tools.neutrino_reference import create_analytic_reference, import_prometheus_sqlite


@pytest.fixture
def photon_shard(tmp_path):
    """Make isolated events, including empty records and repeated configurations."""
    groups = np.repeat(np.arange(10, dtype=np.int64), 2)
    event_ids = np.arange(len(groups), dtype=np.int64)
    n = np.where(event_ids % 4 == 0, 0, 64)
    offsets = np.concatenate(([0], np.cumsum(n))).astype(np.int64)
    path = tmp_path / "photons.h5"
    write_photon_shard(
        path,
        offsets=offsets,
        sensor_index=np.repeat(event_ids % 2, n).astype(np.int32),
        time_ns=np.full(int(n.sum()), 10.0),
        positions_m=np.asarray([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=float),
        live=np.asarray([True, True, False]),
        event_ids=event_ids,
        configuration_ids=groups,
        partition=partition_configurations(groups, seed=7),
        time_edges_ns=np.asarray([0, 25, 50], dtype=float),
        provenance={"kind": "test_fixture", "clock": "fixed"},
    )
    return path, n, groups


def test_partitions_keep_all_replicas_of_each_event_together():
    """An alternate intensity or photon realization cannot leak into another fold."""
    configurations = np.repeat(np.arange(100), 6)
    partition = partition_configurations(configurations, seed=17)
    np.testing.assert_array_equal(partition, partition_configurations(configurations, seed=17))
    assert np.bincount(partition).tolist() == [480, 60, 60]
    assert all(len(set(partition[configurations == index])) == 1 for index in range(100))


def test_event_boundaries_and_empty_observations_are_preserved(photon_shard):
    """Each histogram comes from exactly one source observation, with no concatenation."""
    path, number, _ = photon_shard
    dataset = NeutrinoPhotonDataset(path, "all", training=False)
    assert len(dataset) == len(number)
    for index, expected in enumerate(number):
        item = dataset[index]
        assert int((item["input"] + item["target"]).sum()) == expected
        assert item["input"].dtype == torch.int64
        assert set(item) == {
            "input",
            "target",
            "positions_m",
            "live",
            "p",
            "event_id",
            "outside_window",
        }
    assert dataset[0]["input"].sum() == dataset[0]["target"].sum() == 0


def test_training_resplits_but_validation_remains_frozen(photon_shard):
    """Epoch state affects only training, independently of worker RNG state."""
    path, _, _ = photon_shard
    train = NeutrinoPhotonDataset(path, "all", training=True)
    validation = NeutrinoPhotonDataset(path, "all", training=False)
    before = train[1]["input"].clone()
    frozen = validation[1]["input"].clone()
    train.set_epoch(1)
    validation.set_epoch(90)
    assert not torch.equal(before, train[1]["input"])
    assert torch.equal(frozen, validation[1]["input"])
    assert torch.equal(train[1]["input"], train[1]["input"])


def test_dataset_partitions_have_disjoint_configuration_ids(photon_shard):
    """Training and evaluation constructors select only their persisted partition."""
    path, _, groups = photon_shard
    datasets = [NeutrinoPhotonDataset(path, split) for split in ("train", "validation", "test")]
    selections = [set(groups[dataset.indices]) for dataset in datasets]
    assert not selections[0] & selections[1]
    assert not selections[0] & selections[2]
    assert not selections[1] & selections[2]


def test_fixed_clock_preserves_boundaries_and_reports_outside_photons():
    """Neither the earliest detection nor a held-out photon chooses the clock."""
    sensor = np.asarray([0, 0, 1, 1, 0], dtype=np.int32)
    times = np.asarray([-1.0, 0.0, 25.0, 49.0, 50.0])
    counts, dropped = bin_photons(sensor, times, 2, np.asarray([0.0, 25.0, 50.0]))
    np.testing.assert_array_equal(counts, [[1, 0], [0, 2]])
    assert dropped == 2


@pytest.mark.parametrize("invalid", [np.asarray([1.2]), np.asarray([-1])])
def test_splitter_rejects_fractional_charge_and_negative_counts(invalid):
    """Invalid detector observables cannot be silently floored into photons."""
    with pytest.raises(ValueError, match="integer"):
        split_photon_counts(invalid, 0.5, np.random.default_rng(1))


def test_thinning_conserves_all_counts_and_allows_shared_occupied_bins():
    """Multiple photons in a bin may appear in both sides of a count split."""
    counts = np.asarray([[0, 100, 10000]], dtype=np.int64)
    retained, target = split_photon_counts(counts, 0.5, np.random.default_rng(1))
    np.testing.assert_array_equal(retained + target, counts)
    assert np.all((retained[0, 1:] > 0) & (target[0, 1:] > 0))
    assert retained[0, 0] == target[0, 0] == 0


def test_live_support_excludes_inactive_sensors_from_normalization_and_gradients():
    """An inactive sensor's huge logits must not steal probability mass."""
    logits = torch.tensor([[[0.0, 0.0], [100000.0, 100000.0]]], requires_grad=True)
    live = torch.tensor([True, False])
    target = torch.tensor([[[2, 1], [0, 0]]])
    probability = photon_count_log_probabilities(logits, live).exp()
    torch.testing.assert_close(probability, torch.tensor([[[0.5, 0.5], [0.0, 0.0]]]))
    loss = photon_count_nll(logits, target, live)
    torch.testing.assert_close(loss, torch.tensor(3 * np.log(2), dtype=loss.dtype))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, 1]) == 0


def test_count_likelihood_preserves_photon_weighting_and_empty_targets():
    """An event with three photons contributes three categorical log probabilities."""
    logits = torch.zeros(2, 1, 2, requires_grad=True)
    target = torch.tensor([[[0, 0]], [[1, 2]]])
    loss = photon_count_nll(logits, target, torch.tensor([True]))
    torch.testing.assert_close(loss, torch.tensor(1.5 * np.log(2), dtype=loss.dtype))
    empty = photon_count_nll(logits, torch.zeros_like(target), torch.tensor([True]))
    empty.backward()
    assert empty == 0
    assert torch.count_nonzero(logits.grad) == 0


def test_count_likelihood_rejects_photons_on_inactive_sensors():
    """Invalid support must be detected before forming 0*log(0) terms."""
    with pytest.raises(ValueError, match="outside live"):
        photon_count_nll(
            torch.zeros(1, 2, 2),
            torch.ones(1, 2, 2, dtype=torch.int64),
            torch.tensor([True, False]),
        )


def test_reference_model_is_equivariant_to_sensor_order():
    """Sensor IDs do not create artificial geometric adjacency in the dense model."""
    torch.manual_seed(7)
    model = DensePhotonReference(channels=4).eval()
    counts = torch.poisson(torch.full((2, 3, 16), 0.2))
    positions = torch.randn(2, 3, 3)
    live = torch.ones(2, 3, dtype=torch.bool)
    p = torch.tensor([0.5, 0.5])
    order = torch.tensor([2, 0, 1])
    prediction = model(counts, positions, live, p)
    reordered = model(counts[:, order], positions[:, order], live[:, order], p)
    torch.testing.assert_close(reordered, prediction[:, order])
    loss = photon_count_nll(prediction, counts.to(torch.int64), live)
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_poisson_audit_checks_regenerated_observations_and_fixed_total_control():
    """Correct conditional independence and fixed-total anticorrelation both appear."""
    result = audit_poisson_thinning(seed=42, samples=20000)
    assert result["passed"]
    assert all(
        case["conserves_counts"] and case["multiphoton_bins_present"] for case in result["cases"]
    )
    assert result["fixed_total_negative_control"]["observed_covariance"] == pytest.approx(
        -2.5, abs=0.15
    )


def test_analytic_generation_keeps_replicas_together_and_truth_out_of_training(tmp_path):
    """Generated rates are evaluation companions, never supervised dataset features."""
    path = tmp_path / "analytic.h5"
    config = {
        "seed": 4,
        "events": 60,
        "time_window_ns": [0, 6400],
        "time_bins": 16,
        "analytic": {
            "strings": 2,
            "sensors_per_string": 3,
            "expected_signal_photons": [0, 20],
            "expected_background_photons": 0,
        },
    }
    result = create_analytic_reference(path, config)
    assert result["empty_observations"] >= 30
    assert result["split_conservation_passed"]
    assert result["provenance"]["kind"] == "analytic_poisson_toy_not_neutrino_transport"
    assert path.with_suffix(".evaluation_only.npz").exists()
    with h5py.File(path, "r") as handle:
        assert set(handle) == {"geometry", "photons", "events", "time_edges_ns"}


def test_physical_import_samples_event_table_and_preserves_empty_records(tmp_path):
    """Sampling from photons alone would lose empty events and bias the pilot."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    database = tmp_path / "source.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE mc_truth(event_no INTEGER PRIMARY KEY, initial_state_energy FLOAT)"
        )
        connection.execute(
            "CREATE TABLE photons(sensor_pos_x FLOAT,sensor_pos_y FLOAT,sensor_pos_z FLOAT,"
            "string_id FLOAT,sensor_id FLOAT,t FLOAT,event_no INTEGER)"
        )
        connection.executemany(
            "INSERT INTO mc_truth VALUES (?,?)", [(i, 1000 + i) for i in range(20)]
        )
        connection.executemany(
            "INSERT INTO photons VALUES (?,0,0,?,0,10,?)",
            [(0 if i % 4 == 0 else 10, 0 if i % 4 == 0 else 1, i) for i in range(0, 20, 2)],
        )
    geometry = tmp_path / "geometry.parquet"
    pd.DataFrame(
        {
            "sensor_pos_x": [0.0, 10.0],
            "sensor_pos_y": [0.0, 0.0],
            "sensor_pos_z": [0.0, 0.0],
            "sensor_string_id": [0, 1],
            "sensor_id": [0, 1],
        }
    ).to_parquet(geometry)
    target = tmp_path / "physical.h5"
    result = import_prometheus_sqlite(
        database,
        geometry,
        target,
        {"events": 20, "seed": 42, "time_window_ns": [0, 100], "time_bins": 4},
    )
    assert result["events"] == 20
    assert result["empty_observations"] == 10
    assert result["photons_in_window"] == 10
    assert result["provenance"]["observed_source_sensor_mapping"] == [[0, 0, 0], [1, 0, 1]]
    with h5py.File(target, "r") as handle:
        assert "mc_truth" not in handle
    truth = json.loads(target.with_suffix(".evaluation_only.json").read_text())
    assert len(truth["events"]) == 20
