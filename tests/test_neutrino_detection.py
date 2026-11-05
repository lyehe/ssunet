"""Check count conservation, fair null calibration and grouped recovery evidence."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.ssunet.neutrino_detection import (
    PhotonWindows,
    calibrated_tail_probabilities,
    count_glr,
    cross_shape_score,
    kernel_at_photons,
    normal_kernel,
    sample_windows,
    summarize_detection,
    template_glr,
    wilson_interval,
)
from src.tools.neutrino_detection import FrozenScorer, run_detection, score_windows, validate_config


def test_dense_halves_conserve_empty_and_repeated_observations():
    """Keep empty windows and multiple photons in the same sensor/time bin."""
    windows = PhotonWindows(
        np.array([0, 0, 4]),
        np.array([1, 1, 1, 5]),
        np.array([True, False, True, False]),
        np.array([0.4, 0.7]),
    )
    left, right = windows.dense_batch(0, 2, (2, 3))
    assert len(windows) == 2
    assert not left[0].any()
    assert not right[0].any()
    assert left[1, 0, 1] == 2
    assert right[1, 0, 1] == 1
    assert right[1, 1, 2] == 1
    assert np.array_equal((left + right).sum((1, 2)), np.diff(windows.offsets))


def test_sampling_does_not_drop_poisson_zeros_or_split_fractional_photons():
    """A positive background mean still yields valid empty acquisition windows."""
    number = 5000
    signals = np.full(number, 0.3)
    windows = sample_windows(
        (np.array([0.8, 0.2]) for _ in range(number)), signals, 0.2, 2, np.random.default_rng(42)
    )
    totals = np.diff(windows.offsets)
    assert len(windows) == number
    assert (totals == 0).any()
    assert abs(totals.mean() - 0.5) < 0.05
    assert abs(windows.retained.mean() - 0.5) < 0.05
    assert windows.locations.dtype.kind in "iu"
    assert set(windows.locations) == {0, 1}


def test_poisson_glr_and_template_search_match_direct_likelihood_ratio():
    """Use count multiplicity and the Poisson rate penalty, including empties."""
    np.testing.assert_allclose(
        count_glr([0, 1, 2, 4], 1), [0, 0, 2 * np.log(2) - 1, 4 * np.log(4) - 3]
    )
    templates = np.array([[0.8, 0.2], [0.1, 0.9]])
    locations = np.array([0, 0, 1])
    grid = np.array([0.5, 2.0])
    expected = max(
        sum(np.log((1 / 2 + mu * pdf[location]) / (1 / 2)) for location in locations) - mu
        for mu in grid
        for pdf in templates
    )
    assert template_glr(locations, templates, grid, 1) == pytest.approx(expected)
    assert template_glr(np.empty(0, dtype=int), templates, grid, 1) == 0


def test_vectorized_template_search_matches_sparse_reference():
    """The batched device search includes masked padding and fully empty batches."""
    scorer = FrozenScorer.__new__(FrozenScorer)
    scorer.device = torch.device("cpu")
    scorer.templates = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float64)
    scorer.amplitudes = torch.tensor([0.5, 2.0], dtype=torch.float64)
    scorer.prior = np.ones((1, 2)) / 2
    scorer.config = {"generator": {"background_expected_photons": 1}}
    windows = PhotonWindows(
        np.array([0, 0, 3, 4]),
        np.array([0, 0, 1, 1]),
        np.array([True, False, True, False]),
        np.array([0.1, 0.2, 0.3]),
    )
    expected = [
        0,
        template_glr(np.array([0, 0, 1]), scorer.templates.numpy(), [0.5, 2.0], 1),
        template_glr(np.array([1]), scorer.templates.numpy(), [0.5, 2.0], 1),
    ]
    np.testing.assert_allclose(scorer.template_scores(windows, 0, 3), expected)
    np.testing.assert_array_equal(scorer.template_scores(windows, 0, 1), [0])


def test_uniform_prediction_supplies_no_extra_event_evidence():
    """An always-normalized PDF contributes zero evidence when it equals the null."""
    assert cross_shape_score([0, 0], [1], [0.25], [0.25, 0.25], 4) == 0
    assert cross_shape_score([], [], [], [], 4) == 0
    with pytest.raises(ValueError, match="Every observed photon"):
        cross_shape_score([0, 0], [1], [0.25], [0.25], 4)


def test_kernel_probabilities_are_normalized_on_irregular_geometry():
    """Normalize each source kernel, avoiding density changes at detector edges."""
    prior = np.arange(1, 7, dtype=float).reshape(3, 2) / 21
    spatial = normal_kernel([[0], [1], [5]], 2)
    temporal = normal_kernel([[0], [2]], 1)
    queried = np.arange(6)
    estimate = kernel_at_photons([0, 0, 4], queried, prior, spatial, temporal, 0.4)
    assert estimate.sum() == pytest.approx(1)
    np.testing.assert_allclose(
        kernel_at_photons([], queried, prior, spatial, temporal, 0.4), prior.ravel()
    )


def test_smoothed_calibration_handles_discrete_ties_and_extreme_scores():
    """Identical score distributions receive uniform randomized ranks."""
    randomizers = np.linspace(0.005, 0.995, 100)
    p = calibrated_tail_probabilities(np.ones(100), np.ones(100), randomizers)
    np.testing.assert_allclose(p, randomizers)
    probabilities = calibrated_tail_probabilities([0, 0, 1, 2], [-1, 0, 1, 3], [0.5] * 4)
    np.testing.assert_allclose(probabilities, [0.9, 0.7, 0.4, 0.1])


@pytest.mark.parametrize("bad", [float("nan"), -1.0, 1.0])
def test_invalid_tie_randomizers_are_rejected(bad):
    """Invalid randomizers cannot silently suppress or create false positives."""
    with pytest.raises(ValueError, match="randomizers"):
        calibrated_tail_probabilities([0, 1], [1], [bad])


def test_group_bootstrap_does_not_treat_repeated_source_views_as_new_sources():
    """Repeating every within-source observation leaves source uncertainty unchanged."""
    config = {
        "generator": {"signal_expected_photons": [2]},
        "evaluation": {"bootstrap_draws": 500, "bootstrap_seed": 1, "false_positive_rates": [0.01]},
    }
    truth = {"source_group": np.array([-1, -1, 0, 1]), "expected_signal": np.array([0, 0, 2, 2])}
    p = {"self_split": np.array([0.5, 0.005, 0.001, 0.5]), "count": np.array([0.5, 0.5, 0.5, 0.5])}
    original = summarize_detection(p, truth, config)["0.01"]
    repeated = summarize_detection(
        {key: np.repeat(value, 5) for key, value in p.items()},
        {key: np.repeat(value, 5) for key, value in truth.items()},
        config,
    )["0.01"]
    row = original["count"]["efficiency"][0]
    repeated_row = repeated["count"]["efficiency"][0]
    assert row["difference_interval"] == repeated_row["difference_interval"]
    assert row["model_minus_control"] == 0.5
    assert row["control_missed_model_found"] == 1
    assert row["control_found_model_missed"] == 0
    assert original["self_split"]["false_positives"] == 1


def test_detection_scorer_has_no_truth_input_and_uses_both_recorded_halves():
    """A uniform fake model equals the count control even on repeated photons."""
    windows = PhotonWindows(
        np.array([0, 0, 3]),
        np.array([0, 0, 2]),
        np.array([True, False, True]),
        np.array([0.3, 0.6]),
    )
    scorer = FrozenScorer.__new__(FrozenScorer)
    scorer.config = {
        "generator": {"background_expected_photons": 1},
        "runtime": {
            "batch_size": 2,
            "max_seconds": 1e20,
            "max_process_rss_gib": 1e6,
            "max_gpu_allocated_gib": 1e6,
        },
    }
    scorer.device, scorer.shape = torch.device("cpu"), (2, 2)
    scorer.prior = np.ones((2, 2)) / 4
    scorer.sensor_kernel = scorer.time_kernel = np.eye(2)
    scorer.mixture = {"prior_fraction": 0.5}
    scorer.templates = torch.ones((1, 4), dtype=torch.float64) / 4
    scorer.amplitudes = torch.tensor([1.0, 2.0], dtype=torch.float64)
    seen = []

    def predict(counts):
        seen.append(counts.copy())
        return np.full(counts.shape, 0.25)

    scorer.predict = predict
    result = score_windows(windows, scorer, 0, {"process_rss_gib": 0, "gpu_allocated_gib": 0})
    np.testing.assert_allclose(result["self_split"], result["count"])
    assert len(seen) == 2
    assert sum(int(values.sum()) for values in seen) == len(windows.locations)


def test_wilson_intervals_include_zero_and_one_boundaries():
    """Do not report a zero-width false-alarm interval when no false alarm occurs."""
    assert wilson_interval(0, 100)[1] > 0
    assert wilson_interval(100, 100)[0] < 1
    with pytest.raises(ValueError, match="positive trials"):
        wilson_interval(0, 0)


def test_existing_study_cannot_be_overwritten(tmp_path):
    """A completed or partial output must not be replaced by another test opening."""
    config = yaml.safe_load(Path("config/neutrino_detection.yml").read_text())
    config["output"] = str(tmp_path)
    marker = tmp_path / "test_results.json"
    marker.write_text("preserve first test")
    with pytest.raises(FileExistsError):
        run_detection(config, "cpu")
    assert marker.read_text() == "preserve first test"


def test_frozen_model_split_contract_is_checked_before_opening_data():
    """The fixed feasibility run cannot silently change the trained input exposure."""
    config = yaml.safe_load(Path("config/neutrino_detection.yml").read_text())
    config["generator"]["split_probability"] = 0.25
    with pytest.raises(ValueError, match="registered split"):
        validate_config(config)
