"""Assess frozen self-split distributions with separate labelled energy/direction probes."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from ..ssunet.datasets.neutrino import shard_sha256
from ..ssunet.models.neutrino_dense import DensePhotonReference
from ..ssunet.photon_reference import photon_count_log_probabilities
from .neutrino_ablations import mechanism_dataset
from .neutrino_prepare import write_json
from .neutrino_study import _bank_from_selection, _training_reference, verify_analysis_lock


def distribution_features(probability, retained_total, coarse_bins=64):
    """Give each method the same sensor/time marginals, moments and input total."""
    probability = np.asarray(probability, dtype=np.float64)
    batch, sensors, bins = probability.shape
    if bins % coarse_bins:
        message = "Probe time bins must divide the fixed acquisition bins"
        raise ValueError(message)
    sensor = probability.sum(2)
    temporal = probability.reshape(batch, sensors, coarse_bins, bins // coarse_bins).sum((1, 3))
    coordinate = np.linspace(-1, 1, bins)
    first_moment = (probability * coordinate).sum(2) / np.maximum(sensor, 1e-12)
    total = np.asarray(retained_total)
    return np.concatenate(
        [sensor, temporal, first_moment, np.log1p(total)[:, None], (total == 0)[:, None]], axis=1
    )


def fit_ridge(features, targets, penalty):
    """Fit a standardized ridge probe using labelled training records only."""
    mean, scale = features.mean(0), features.std(0)
    scale = np.maximum(scale, 1e-6)
    x = (features - mean) / scale
    y_mean = targets.mean(0)
    weights = np.linalg.solve(
        x.T @ x / len(x) + penalty * np.eye(x.shape[1]), x.T @ (targets - y_mean) / len(x)
    )
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "target_mean": y_mean,
        "weights": weights,
        "penalty": penalty,
    }


def predict_ridge(probe, features):
    """Apply a frozen labelled probe without updating preprocessing statistics."""
    return ((features - probe["feature_mean"]) / probe["feature_scale"]) @ probe["weights"] + probe[
        "target_mean"
    ]


def angular_errors(predictions, targets):
    """Return spherical angular error in degrees, handling a zero vector explicitly."""
    norm = np.linalg.norm(predictions, axis=1, keepdims=True)
    unit = predictions / np.maximum(norm, 1e-12)
    unit = np.where(norm > 1e-12, unit, np.array([0.0, 0.0, 1.0]))
    return np.rad2deg(np.arccos(np.clip((unit * targets).sum(1), -1, 1)))


def _labels(main, ids):
    """Read MC labels for the explicitly labelled probe, never for pretraining."""
    path = Path(main["data"]["output"]) / "source_truth.evaluation_only.parquet"
    columns = ["event_no", "initial_state_energy", "initial_state_zenith", "initial_state_azimuth"]
    frame = (
        pd.read_parquet(path, columns=columns, filters=[("event_no", "in", ids.tolist())])
        .set_index("event_no")
        .loc[ids]
    )
    energy = frame.initial_state_energy.to_numpy()
    zenith, azimuth = frame.initial_state_zenith.to_numpy(), frame.initial_state_azimuth.to_numpy()
    if not (
        np.isfinite(energy).all()
        and np.all(energy > 0)
        and np.isfinite(zenith).all()
        and np.isfinite(azimuth).all()
    ):
        message = "Probe labels must be finite with positive stored energy"
        raise ValueError(message)
    direction = np.column_stack(
        [np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)]
    )
    return {"energy": np.log10(energy)[:, None], "direction": direction}


@torch.inference_mode()
def _extract(config, partition, models, bank, device):
    """Extract A-only summaries under identical feature dimensions for all methods."""
    dataset = mechanism_dataset(config, partition)
    ids = dataset.event_ids[dataset.indices]
    groups = dataset.configuration_ids[dataset.indices]
    features = {"raw": [], "baseline": [], **{f"model_seed_{seed}": [] for seed in models}}
    totals = []
    for batch in DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False):
        a = batch["input"].to(device)
        total = a.sum((1, 2))
        uniform = torch.full_like(a, 1 / (a.shape[1] * a.shape[2]), dtype=torch.float32)
        raw_pdf = torch.where(
            total[:, None, None] > 0, a / total[:, None, None].clamp_min(1), uniform
        )
        baseline = bank.probabilities(a)[bank.winner]
        values = {"raw": raw_pdf, "baseline": baseline}
        for seed, model in models.items():
            logits = model(
                a, batch["positions_m"].to(device), batch["live"].to(device), batch["p"].to(device)
            )
            values[f"model_seed_{seed}"] = photon_count_log_probabilities(
                logits, batch["live"].to(device)
            ).exp()
        for method, probability in values.items():
            features[method].append(
                distribution_features(
                    probability.cpu().numpy(),
                    total.cpu().numpy(),
                    config["downstream"]["feature_time_bins"],
                )
            )
        totals.extend(total.cpu().tolist())
    dataset.close()
    return (
        {method: np.concatenate(values) for method, values in features.items()},
        ids,
        groups,
        np.asarray(totals),
    )


def paired_mean_interval(delta, groups, *, seed=20261025, draws=10000, chunk=64):
    """Resample source configurations for a labelled per-event error contrast."""
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=delta)
    sizes = np.bincount(inverse)
    generator = np.random.default_rng(seed)
    values = []
    for first in range(0, draws, chunk):
        selected = generator.integers(len(unique), size=(min(chunk, draws - first), len(unique)))
        values.extend((sums[selected].sum(1) / sizes[selected].sum(1)).tolist())
    return {
        "mean_error_difference": float(np.mean(delta)),
        "descriptive_95_percent_interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "resampling_unit": "persisted_source_configuration",
        "model_seed_or_selection_uncertainty_included": False,
    }


def run_probe(config, device):
    """Select labelled linear probes on validation, then score the untouched probe test."""
    root = Path(config["data"]["output"]) / "downstream_probe"
    root.mkdir(exist_ok=False)
    main = yaml.safe_load(Path(config["main_study_config"]).read_text())
    main_root = Path(main["data"]["output"])
    if not (main_root / "test_results.json").exists():
        message = "Complete the primary self-supervised test before labelled downstream work"
        raise ValueError(message)
    lock = verify_analysis_lock(main)
    device = torch.device(device)
    torch.set_num_threads(config["training"]["cpu_threads"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    models = {}
    for seed, checkpoint in lock["checkpoints"].items():
        model = DensePhotonReference(main["model"]["channels"]).to(device)
        model.load_state_dict(
            torch.load(checkpoint["path"], map_location="cpu", weights_only=True)["state_dict"]
        )
        model.eval()
        models[int(seed)] = model
    training, reference = _training_reference(main)
    bank = _bank_from_selection(main, training, reference, device)
    started = time.perf_counter()
    x_train, ids_train, _, _ = _extract(config, "train", models, bank, device)
    x_val, ids_val, _, _ = _extract(config, "validation", models, bank, device)
    y_train, y_val = _labels(main, ids_train), _labels(main, ids_val)
    probes, validation = {}, {}
    for method in x_train:
        probes[method], validation[method] = {}, {}
        for task in ("energy", "direction"):
            candidates = []
            fitted = []
            for penalty in config["downstream"]["ridge_penalties"]:
                probe = fit_ridge(x_train[method], y_train[task], penalty)
                predicted = predict_ridge(probe, x_val[method])
                score = (
                    float(np.mean(np.abs(predicted - y_val[task])))
                    if task == "energy"
                    else float(angular_errors(predicted, y_val[task]).mean())
                )
                candidates.append({"penalty": penalty, "validation_error": score})
                fitted.append(probe)
            chosen = int(np.argmin([row["validation_error"] for row in candidates]))
            probes[method][task] = fitted[chosen]
            validation[method][task] = {
                "candidates": candidates,
                "selected_penalty": candidates[chosen]["penalty"],
            }
    write_json(
        root / "analysis_lock.json",
        {
            "configuration": config["downstream"],
            "validation_selection": validation,
            "pretrained_lock_sha256": shard_sha256(main_root / "analysis_lock.json"),
            "probe_cohort_sha256": shard_sha256(
                Path(config["data"]["output"]) / "cohort_manifest.parquet"
            ),
            "code_sha256": shard_sha256(Path(__file__)),
            "labels_are_only_for_probe": True,
            "feature_dimension": int(next(iter(x_train.values())).shape[1]),
            "self_supervised_models_frozen": True,
        },
    )
    write_json(
        root / "test_opened.json",
        {"probe_analysis_lock_sha256": shard_sha256(root / "analysis_lock.json")},
    )
    x_test, ids, groups, totals = _extract(config, "test", models, bank, device)
    y_test = _labels(main, ids)
    errors = {"energy": {}, "direction": {}}
    for method in x_test:
        energy = predict_ridge(probes[method]["energy"], x_test[method])
        direction = predict_ridge(probes[method]["direction"], x_test[method])
        errors["energy"][method] = np.abs(energy[:, 0] - y_test["energy"][:, 0])
        errors["direction"][method] = angular_errors(direction, y_test["direction"])
    report = {
        "scope": (
            "independent_labelled_linear_probe; curated_PONE_MC; "
            "not_real_detector_physics_performance"
        ),
        "independent_test_events": len(ids),
        "independent_source_groups": len(np.unique(groups)),
        "self_supervised_test_not_reused": True,
        "feature_dimension": int(next(iter(x_test.values())).shape[1]),
        "metrics": {},
        "strata": {},
        "elapsed_seconds": time.perf_counter() - started,
        "energy_units": "log10 of stored source energy; no archive unit certification inferred",
        "direction_label": "unit vector from stored initial zenith/azimuth in radians",
        "training_labels_do_not_enter_self_supervised_pretraining": True,
    }
    for task, values in errors.items():
        values["mean_seed_model"] = np.mean(
            [values[f"model_seed_{seed}"] for seed in models], axis=0
        )
        report["metrics"][task] = {
            "mean_errors": {method: float(error.mean()) for method, error in values.items()},
            "model_minus_raw": paired_mean_interval(
                values["mean_seed_model"] - values["raw"], groups
            ),
            "model_minus_baseline": paired_mean_interval(
                values["mean_seed_model"] - values["baseline"], groups
            ),
        }
        report["strata"][task] = {}
        for name, mask in (
            ("retained_0", totals == 0),
            ("retained_1_to_10", (totals >= 1) & (totals <= 10)),
            ("retained_above_10", totals > 10),
        ):
            report["strata"][task][name] = {
                "events": int(mask.sum()),
                "mean_errors": {
                    method: float(error[mask].mean()) if mask.any() else None
                    for method, error in values.items()
                },
            }
    np.savez_compressed(
        root / "test_errors.npz",
        event_ids=ids,
        configuration_ids=groups,
        retained_totals=totals,
        **{
            f"{task}_{method}": error
            for task, values in errors.items()
            for method, error in values.items()
        },
    )
    write_json(root / "test_results.json", report)
    training.close()


def main():
    """Run the finite independent downstream utility comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/neutrino_ablations.yml"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_probe(yaml.safe_load(args.config.read_text()), args.device)


if __name__ == "__main__":
    main()
