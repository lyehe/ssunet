"""Verify permanent photon isolation, count likelihoods and resumable joint training."""

import copy
import math

import numpy as np
import pytest
import torch

from src.ssunet.confocal_trial import (
    evaluate_baselines,
    prepare_training_pool,
    score_log_probabilities,
    summarize_scores,
    thin_frame,
    training_batch,
    validation_cases,
)
from src.ssunet.models.confocal_multichannel import MultichannelConfocalGAP
from src.tools import confocal_joint_train


def _config():
    return {
        "training": {
            "patch_tzyx": [2, 2, 2, 2],
            "batch_size": 2,
            "accumulation_steps": 1,
            "photon_split_seed": 14,
            "split_probability": 0.5,
            "learning_rate": 0.001,
            "gradient_clip": 0.5,
            "steps": 4,
            "checkpoint_every": 1,
        },
        "evaluation": {
            "split_seed": 17,
            "phases": {"early": 0, "late": 2},
            "spatial_starts": [[0, 0, 0]],
            "uniform_mixture": 0.01,
            "gaussian_spatial_sigma_zyx": [0.5, 1, 1],
            "gaussian_temporal_sigma": 1,
            "every_steps": 2,
        },
    }


def _raw_and_pool():
    raw = (np.arange(2 * 4**4).reshape(2, 4, 4, 4, 4) % 8).astype(np.uint8)
    pool = np.stack(
        [
            np.stack(
                [
                    thin_frame(frame, 0.8, seed=5, channel=c, time=t)
                    for t, frame in enumerate(channel)
                ]
            )
            for c, channel in enumerate(raw)
        ]
    )
    return raw, pool


def test_training_pool_conserves_photons_and_rejects_cache_corruption(tmp_path):
    """No evaluation-only photon can appear in either side of a training split."""
    raw, expected = _raw_and_pool()
    destination = tmp_path / "pool.npy"
    pool, manifest = prepare_training_pool(
        raw, destination, {"training_probability": 0.8, "seed": 5}, "raw-sha"
    )
    np.testing.assert_array_equal(pool, expected)
    assert np.all(pool <= raw)
    assert np.any(pool < raw)
    again, reused = prepare_training_pool(
        raw, destination, {"training_probability": 0.8, "seed": 5}, "raw-sha"
    )
    np.testing.assert_array_equal(again, pool)
    assert reused == manifest
    altered = np.load(destination, mmap_mode="r+")
    altered[0, 0, 0, 0, 0] = 255
    altered.flush()
    with pytest.raises(ValueError, match="identity or content"):
        prepare_training_pool(raw, destination, {"training_probability": 0.8, "seed": 5}, "raw-sha")


def test_fixed_evaluation_uses_only_photons_excluded_from_all_training(tmp_path):
    """Targets are the permanent complement, rather than a new split of reused training photons."""
    raw, pool = _raw_and_pool()
    cases = validation_cases(raw, pool, _config())
    for case in cases:
        start = case["start_tzyx"]
        select = (slice(None), *(slice(a, a + 2) for a in start))
        training = pool[select]
        np.testing.assert_array_equal(case["target"][0].numpy() + training, case["observed"])
        assert np.all(case["input"][0].numpy() <= training)
    corrupted = pool.copy()
    corrupted[0, 0, 0, 0, 0] = 255
    with pytest.raises(ValueError, match="subset"):
        validation_cases(raw, corrupted, _config())


def test_joint_training_draws_do_not_change_with_microbatch_grouping():
    """Effective batch size and example addresses determine the split, not physical grouping."""
    _, pool = _raw_and_pool()
    large = _config()
    large["training"].update(batch_size=4, accumulation_steps=1)
    small = copy.deepcopy(large)
    small["training"].update(batch_size=1, accumulation_steps=4)
    a, b = training_batch(pool, large, step=3, microstep=0)
    pieces = [training_batch(pool, small, step=3, microstep=i) for i in range(4)]
    torch.testing.assert_close(a, torch.cat([part[0] for part in pieces]), rtol=0, atol=0)
    torch.testing.assert_close(b, torch.cat([part[1] for part in pieces]), rtol=0, atol=0)


def test_scores_use_photon_weighting_and_keep_empty_channels_explicit():
    """A zero-photon case has no accuracy score and cannot cause NaN in phase summaries."""
    logq = torch.full((1, 2, 2, 2, 2, 2), -math.log(16), dtype=torch.float64)
    target = torch.zeros_like(logq, dtype=torch.int64)
    target[:, 1, 0, 0, 0, 0] = 7
    row = score_log_probabilities(
        logq, target, {"index": 0, "phase": "late", "start_tzyx": [0] * 4}
    )
    summary = summarize_scores([row])
    assert row["channels"][0]["photon_nll"] is None
    assert summary["target_photons"] == 7
    assert summary["empty_cases"] == 1
    assert summary["photon_nll"] == pytest.approx(math.log(16), abs=1e-12)
    assert summary["channel_mean_nll"] == pytest.approx(math.log(16), abs=1e-12)


def test_fixed_gaussian_controls_are_finite_for_empty_and_sparse_inputs():
    """The input-only controls retain unit probability mass without target-driven tuning."""
    raw, pool = _raw_and_pool()
    cases = validation_cases(raw, pool, _config())
    cases[0]["input"].zero_()
    controls = evaluate_baselines(cases, _config())
    for score in controls.values():
        assert math.isfinite(score["photon_nll"])
        assert score["max_probability_mass_error"] < 1e-6
        assert score["cases"][0]["channels"][1]["photon_nll"] == pytest.approx(math.log(16))


def test_interrupted_joint_training_restores_the_exact_optimizer_trajectory(tmp_path, monkeypatch):
    """A resumed CPU fit matches uninterrupted weights and losses at the fixed endpoint."""
    torch.set_num_threads(1)
    raw, pool = _raw_and_pool()
    config = _config()
    cases = validation_cases(raw, pool, config)
    identity = {"source_hashes": {"unit": "frozen"}, "config": config}
    monkeypatch.setattr(confocal_joint_train, "source_hashes", lambda: identity["source_hashes"])

    def new_model():
        torch.manual_seed(34)
        return MultichannelConfocalGAP(
            channels=2, dimensions=4, base_channels=2, depth=2, encoding_levels=2, backend="torch"
        )

    device = torch.device("cpu")
    uninterrupted = new_model()
    expected = confocal_joint_train.run_training(
        uninterrupted, pool, cases, config, tmp_path / "whole", identity, device
    )
    partial = new_model()
    step_function = confocal_joint_train.optimizer_step

    def interrupt_at_three(*args):
        if args[-1] == 3:
            message = "simulated interruption"
            raise RuntimeError(message)
        return step_function(*args)

    monkeypatch.setattr(confocal_joint_train, "optimizer_step", interrupt_at_three)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        confocal_joint_train.run_training(
            partial, pool, cases, config, tmp_path / "resumed", identity, device
        )
    monkeypatch.setattr(confocal_joint_train, "optimizer_step", step_function)
    resumed = new_model()
    actual = confocal_joint_train.run_training(
        resumed, pool, cases, config, tmp_path / "resumed", identity, device, resume=True
    )
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)
    assert actual["losses"] == expected["losses"]
    assert actual["evaluations"] == expected["evaluations"]
