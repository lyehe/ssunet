"""Check that GAP diagnostic contrasts really weight events instead of photons."""

import numpy as np
import pytest


def test_event_contrasts_exclude_empty_targets_and_keep_grouped_event_weighting():
    """A hundred-photon target gets one vote; empty and unmatched cases stay excluded."""
    pytest.importorskip("torch_convnd")
    from src.ssunet.neutrino_evaluation import retained_count_strata
    from src.tools.neutrino_gap4d_report import event_contrasts

    controls = np.array([1.0, 100.0, 0.0, 3.0])
    retained = np.array([1, 4, 2, 0])
    saved = {
        "event_ids": np.arange(4),
        "configuration_ids": np.array([7, 7, 8, 9]),
        "target_totals": np.array([1, 100, 0, 3]),
        "retained_totals": retained,
        "conv4d": np.array([[3.0, 100.0, 0.0, 12.0]] * 2),
        "dense_reference": np.stack([controls, controls]),
        "full_time_reference": np.stack([controls, controls]),
        "fixed_mixture": controls,
        "previously_tuned_mixture": controls,
        "conv4d_donor": np.stack([controls, controls]),
    }
    coverage = dict.fromkeys(retained_count_strata(retained), 1.0)
    coverage["all"] = 0.5
    config = {"evaluation": {"bootstrap_seed": 17, "bootstrap_draws": 1000}}
    result = event_contrasts(saved, config, coverage)
    assert result["full_time_reference"]["all"][
        "model_minus_control_mean_event_cross_entropy"
    ] == pytest.approx(5 / 3)
    low = result["full_time_reference"]["retained_1_to_10"]
    assert low["model_minus_control_mean_event_cross_entropy"] == pytest.approx(1.0)
    assert low["descriptive_95_percent_interval"] == pytest.approx([1.0, 1.0])
    assert low["resampling_unit"] == "persisted_configuration_id"
    assert result["conv4d_donor"]["all"] is None
    assert result["conv4d_donor"]["retained_1_to_10"] is not None
    assert result["fixed_mixture"]["retained_above_30"] is None
