"""Audit a completed probe from saved observations/features without refitting heads."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..ssunet.datasets.neutrino import shard_sha256
from ..ssunet.neutrino_detection import PhotonWindows, calibrated_tail_probabilities, count_glr
from ..ssunet.neutrino_detection_probe import HEAD_ARMS, feature_design, predict_head
from .neutrino_detection import FrozenScorer
from .neutrino_detection_probe import (
    FrozenFeatureExtractor,
    _head_scores,
    extract_features,
    verify_probe_inputs,
)
from .neutrino_prepare import write_json


def _arrays(path):
    with np.load(path, allow_pickle=False) as handle:
        result = {}
        for key in handle.files:
            value = handle[key]
            result[key] = value.item() if value.ndim == 0 else value
        return result


def _subset(observed, selected):
    offsets, locations, retained = [0], [], []
    for index in selected:
        row = slice(observed.offsets[index], observed.offsets[index + 1])
        locations.append(observed.locations[row])
        retained.append(observed.retained[row])
        offsets.append(offsets[-1] + len(locations[-1]))
    return PhotonWindows(
        np.asarray(offsets),
        np.concatenate(locations),
        np.concatenate(retained),
        observed.randomizers[selected],
    )


def _check_observations(observed, support_size):
    """Reject malformed offsets or photons outside the declared sensor/time support."""
    if (
        observed.offsets[0] != 0
        or observed.offsets[-1] != len(observed.locations)
        or len(observed.locations) != len(observed.retained)
        or not (np.diff(observed.offsets) >= 0).all()
        or not ((observed.locations >= 0) & (observed.locations < support_size)).all()
    ):
        message = "Persisted observations fail the photon-conservation/support audit"
        raise ValueError(message)


def audit_probe(root, device):
    """Check locks, train-only scalers, score replay and batch-level numerical stability."""
    result = json.loads((root / "test_results.json").read_text())
    config = json.loads((root / "config.json").read_text())
    identity = result["identity"]
    verify_probe_inputs(root, identity)
    for name, digest in identity["code_sha256"].items():
        if shard_sha256(name) != digest or shard_sha256(root / "source_snapshot" / name) != digest:
            message = f"The original computation source changed: {name}"
            raise ValueError(message)
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    heads = {arm: _arrays(root / f"head_{arm}.npz") for arm in HEAD_ARMS}
    training = _arrays(root / "train/features.npz")
    scaler_difference = {}
    for arm, head in heads.items():
        matrix = feature_design(
            training["base"], None if arm == "raw_prior" else training[arm], head["extra_dimension"]
        ).astype(np.float64)
        mean = matrix.mean(0)
        scale = np.maximum(matrix.std(0), config["head"]["feature_scale_floor"])
        np.testing.assert_allclose(head["mean"], mean, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(head["scale"], scale, atol=1e-10, rtol=1e-10)
        scaler_difference[arm] = float(np.max(np.abs(head["mean"] - mean)))
    del training, matrix
    old = yaml.safe_load(Path(config["frozen_config"]).read_text())
    prior_config = yaml.safe_load(Path(old["pretraining_config"]).read_text())
    with np.load(
        Path(prior_config["data"]["output"]) / "observations.evaluation_only.npz",
        allow_pickle=False,
    ) as data:
        seen = {tuple(source) for source in data["latent_source_position_m"]}
    with np.load(Path(old["output"]) / "evaluation_only_truth.npz", allow_pickle=False) as data:
        seen.update(tuple(source) for source in data["sources_m"])
    for fold in ("train", "validation", "test"):
        sources = _arrays(root / fold / "injection_labels.npz")["sources_m"]
        if any(tuple(source) in seen for source in sources):
            message = "Source overlap found across folds or earlier studies"
            raise ValueError(message)
        seen.update(tuple(source) for source in sources)
    with np.load(root / "frozen_reference.npz", allow_pickle=False) as handle:
        prior = handle["prior"].copy()
    calibration = _arrays(root / "calibration_scores.npz")
    probabilities = _arrays(root / "test_tail_probabilities.npz")
    stored = _arrays(root / "test_scores.npz")
    observed = PhotonWindows(**_arrays(root / "test/observations.npz"))
    _check_observations(observed, prior.size)
    np.testing.assert_array_equal(
        stored["count"],
        count_glr(np.diff(observed.offsets), config["generator"]["background_expected_photons"]),
    )
    for name, values in stored.items():
        calculated = calibrated_tail_probabilities(calibration[name], values, observed.randomizers)
        np.testing.assert_array_equal(calculated, probabilities[name])
    cached = _arrays(root / "test/features.npz")
    cached_score_differences = {}
    for arm, head in heads.items():
        score = predict_head(
            head,
            cached["base"],
            None if arm == "raw_prior" else cached[arm],
            batch_size=config["runtime"]["head_batch_size"],
        )
        np.testing.assert_allclose(score, stored[arm], atol=1e-10, rtol=1e-10)
        cached_score_differences[arm] = float(np.max(np.abs(score - stored[arm])))
    background = config["generator"]["test"]["background_windows"]
    selected = np.concatenate(
        [np.arange(32), background + np.arange(64), np.arange(len(observed) - 32, len(observed))]
    )
    small_cached = {key: value[selected].copy() for key, value in cached.items()}
    del cached
    scorer = FrozenScorer(old, device)
    extractor = FrozenFeatureExtractor(scorer, config["features"])
    replay, _ = extract_features(
        _subset(observed, selected), extractor, config, "audit", lambda: None
    )
    feature_difference = {}
    for key, value in replay.items():
        np.testing.assert_allclose(value, small_cached[key], atol=2e-5, rtol=1e-5)
        feature_difference[key] = float(np.max(np.abs(value - small_cached[key])))
    replayed_scores = _head_scores(heads, replay, config)
    replay_differences, changes = {}, {}
    alpha = config["evaluation"]["primary_false_positive_rate"]
    for arm, score in replayed_scores.items():
        difference = float(np.max(np.abs(score - stored[arm][selected])))
        replay_differences[arm] = difference
        p = calibrated_tail_probabilities(calibration[arm], score, observed.randomizers[selected])
        changes[arm] = int(((p <= alpha) != (probabilities[arm][selected] <= alpha)).sum())
    report = {
        "status": "passed"
        if max(replay_differences.values()) <= 0.001 and not any(changes.values())
        else "review_needed",
        "frozen_code_heads_calibration_verified": True,
        "source_overlap": 0,
        "test_windows": len(observed),
        "train_only_scaler_max_mean_difference": scaler_difference,
        "cached_head_score_max_difference": cached_score_differences,
        "all_tail_probabilities_reproduced_exactly": True,
        "replayed_case_ids": selected.tolist(),
        "replayed_feature_max_difference": feature_difference,
        "replayed_logit_max_difference": replay_differences,
        "replayed_primary_decision_changes": changes,
    }
    write_json(root / "artifact_audit.json", report)
    return report


def main():
    """Audit only a completed probe, preserving every fitted model and test decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("data/neutrino/detection_probe_v1"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    started = time.perf_counter()
    report = audit_probe(args.study, args.device)
    report.pop("replayed_case_ids")
    print(json.dumps({**report, "audit_seconds": time.perf_counter() - started}, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
