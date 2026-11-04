"""Scientific-score and checkpoint checks for the longer photon pilot."""

import copy

import numpy as np
import pytest
import torch

from src.ssunet.datasets.neutrino import (
    NeutrinoPhotonDataset,
    partition_configurations,
    shard_sha256,
    write_photon_shard,
)
from src.ssunet.models.neutrino_dense import DensePhotonReference
from src.ssunet.neutrino_evaluation import (
    FrozenPhotonView,
    fit_pilot_baselines,
    freeze_photon_view,
    matched_donor_indices,
    model_event_nll,
    paired_photon_interval,
    photon_score_summary,
    probability_nll,
)
from src.tools.neutrino_pilot import run_pilot


@pytest.fixture
def pilot_config(tmp_path):
    """Use grouped observations with empty events and an inactive sensor."""
    path = tmp_path / "pilot.h5"
    identifiers = np.arange(80, dtype=np.int64)
    groups = identifiers // 2
    number = np.where(identifiers % 4 == 0, 0, 12)
    write_photon_shard(
        path,
        offsets=np.concatenate(([0], np.cumsum(number))).astype(np.int64),
        sensor_index=np.repeat(identifiers % 2, number).astype(np.int32),
        time_ns=np.full(int(number.sum()), 10.0),
        positions_m=np.asarray([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=float),
        live=np.asarray([True, True, False]),
        event_ids=identifiers,
        configuration_ids=groups,
        partition=partition_configurations(groups, seed=7),
        time_edges_ns=np.linspace(0, 200, 9),
        provenance={"kind": "test_fixture"},
    )
    return {
        "dataset": str(path),
        "dataset_sha256": shard_sha256(path),
        "split_probability": 0.5,
        "photon_split_seed": 42,
        "model_channels": 4,
        "training": {
            "seeds": [42],
            "steps": 3,
            "batch_size": 8,
            "learning_rate": 0.001,
            "gradient_clip": 1.0,
            "cpu_threads": 1,
            "max_seconds_per_seed": 30,
            "memory_target_gib": 12,
        },
        "evaluation": {
            "steps": [0, 1, 3],
            "fixed_training_events": 8,
            "training_subset_seed": 17,
            "donor_seed": 914,
            "donor_draws": 2,
            "bootstrap_seed": 17,
            "bootstrap_draws": 128,
        },
        "baselines": {
            "original_prior_pseudocount": 0.5,
            "tuned_prior_time_bandwidth_ns": 200,
            "tuned_prior_pseudocount": 0.0001,
            "tuned_mixture_prior_fraction": 0.25,
        },
    }


def test_scores_preserve_photon_weighting_and_bootstrap_whole_groups():
    """Neither scores nor bootstrap ratios average normalized event losses."""
    view = FrozenPhotonView(
        inputs=torch.tensor([[[0, 0]], [[1, 0]], [[2, 0]]]),
        targets=torch.tensor([[[0, 0]], [[1, 0]], [[0, 3]]]),
        positions_m=torch.zeros(1, 3),
        live=torch.tensor([True]),
        p=0.5,
        event_ids=np.arange(3),
        configuration_ids=np.asarray([0, 1, 1]),
    )
    scores = photon_score_summary(np.asarray([0.0, 1.0, 9.0]), view)
    assert scores["all"]["nats_per_photon"] == 2.5
    assert scores["all"]["held_out_photons"] == 4
    assert scores["retained_0"]["nats_per_photon"] is None
    interval = paired_photon_interval(
        2 * view.target_totals, view, view.strata["all"], seed=9, draws=1000
    )
    assert interval["model_minus_control_nats_per_photon"] == 2.0
    assert interval["descriptive_95_percent_interval"] == [2.0, 2.0]
    assert interval["valid_bootstrap_draws"] < 1000


def test_probability_score_preserves_zero_support_and_empty_targets():
    """Zero mass on inactive bins is safe; fractional pseudo-photons are rejected."""
    probability = torch.tensor([[[0.25, 0.75], [0.0, 0.0]], [[0.25, 0.75], [0.0, 0.0]]])
    target = torch.tensor([[[1, 2], [0, 0]], [[0, 0], [0, 0]]])
    value = probability_nll(probability, target)
    np.testing.assert_allclose(value, [-np.log(0.25) - 2 * np.log(0.75), 0], rtol=1e-6)
    with pytest.raises(ValueError, match="integer"):
        probability_nll(probability, target.float())
    target[0, 1, 0] = 1
    with pytest.raises(ValueError, match="support"):
        probability_nll(probability, target)


def test_donor_choices_depend_only_on_retained_counts_and_are_reproducible():
    """Count ties randomize among valid donors; nearest unmatched counts are explicit."""
    donor = np.asarray([1, 1, 2, 4])
    retained = np.asarray([1, 2, 3])
    selected = matched_donor_indices(donor, retained, seed=9, draws=20)
    np.testing.assert_array_equal(
        selected, matched_donor_indices(donor, retained, seed=9, draws=20)
    )
    assert set(selected[:, 0]) == {0, 1}
    assert np.all(selected[:, 1] == 2)
    assert set(selected[:, 2]) == {2, 3}


def test_frozen_views_and_prior_controls_keep_empty_events(pilot_config):
    """Controls remain finite on physical support and frozen across epoch changes."""
    path = pilot_config["dataset"]
    training = NeutrinoPhotonDataset(path, "train", training=False)
    dataset = NeutrinoPhotonDataset(path, "validation")
    before = freeze_photon_view(dataset)
    dataset.set_epoch(7)
    after = freeze_photon_view(dataset)
    torch.testing.assert_close(before.inputs, after.inputs)
    scores = fit_pilot_baselines(training, before, pilot_config["baselines"])
    assert all(np.isfinite(value).all() for value in scores.values())
    assert all(np.all(value[before.target_totals == 0] == 0) for value in scores.values())
    empty = before.inputs.sum(dim=(1, 2)).numpy() == 0
    np.testing.assert_array_equal(
        scores["fixed_mixture"][empty], scores["original_training_prior"][empty]
    )
    with pytest.raises(ValueError, match="frozen"):
        freeze_photon_view(NeutrinoPhotonDataset(path, "train"))


def test_bounded_training_saves_reloadable_checkpoints_and_data_identity(pilot_config, tmp_path):
    """Reloaded weights reproduce validation likelihood with persisted model settings."""
    output = tmp_path / "run"
    result = run_pilot(pilot_config, output, "cpu")
    assert not result["test_partition_scored"]
    assert not result["truth_sidecars_used"]
    run = result["runs"][0]
    assert run["optimizer_steps"] == 3
    assert run["completed_fixed_schedule"]
    assert [row["optimizer_step"] for row in run["learning_curve"]] == [0, 1, 3]
    checkpoint = torch.load(output / "seed_42" / "step_000003.pt", weights_only=True)
    assert checkpoint["dataset_sha256"] == pilot_config["dataset_sha256"]
    assert checkpoint["optimizer_state_dict"]["state"]
    model = DensePhotonReference(checkpoint["model_channels"])
    model.load_state_dict(checkpoint["state_dict"])
    view = freeze_photon_view(NeutrinoPhotonDataset(pilot_config["dataset"], "validation"))
    scores = photon_score_summary(model_event_nll(model, view, "cpu", 8), view)
    expected = run["final"]["strata"]["all"]["scores"]["model"]["nats_per_photon"]
    assert scores["all"]["nats_per_photon"] == pytest.approx(expected, abs=1e-6)
    with pytest.raises(FileExistsError):
        run_pilot(pilot_config, output, "cpu")


def test_changed_observation_hash_stops_before_creating_output(pilot_config, tmp_path):
    """A run cannot silently train against data different from its recorded identity."""
    altered = copy.deepcopy(pilot_config)
    altered["dataset_sha256"] = "0" * 64
    output = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="hash"):
        run_pilot(altered, output, "cpu")
    assert not output.exists()
