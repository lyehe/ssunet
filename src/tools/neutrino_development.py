"""Diagnose detector selection and timing using training/validation data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from ..ssunet.datasets.neutrino import shard_sha256
from ..ssunet.models.neutrino_dense import DensePhotonReference
from ..ssunet.neutrino_detection import PhotonWindows
from ..ssunet.neutrino_detection_probe import (
    HEAD_ARMS,
    encoder_features_tensor,
    feature_design,
    observation_features,
    predict_head,
)
from ..ssunet.neutrino_development import (
    DevelopmentStore,
    diagnostic_head_path,
    pdf_hidden_indices,
    permute_sensor_time_pairings,
    predict_timing,
    ridge_timing_path,
    validation_tail_sensitivity,
)
from .neutrino_analytic_study import analytic_location_pdf
from .neutrino_detection import _resource_guard
from .neutrino_detection_probe import FrozenFeatureExtractor
from .neutrino_detection_probe_audit import _subset
from .neutrino_development_visualize import create_development_report
from .neutrino_prepare import write_json

LOGGER = logging.getLogger(__name__)


def _head(store, arm):
    return store.arrays(f"head_{arm}.npz")


def selection_diagnosis(store, config, source_config, development, truth, observed, output, guard):
    """Reproduce the old grid and compare two metrics on validation, never test."""
    choices = store.json("head_selection.json")
    y_train = (truth["train"]["expected_signal"] > 0).astype(float)
    y_validation = (truth["validation"]["expected_signal"] > 0).astype(float)
    result = {}
    for arm in HEAD_ARMS:
        candidates = []
        saved_scores = predict_head(
            _head(store, arm),
            development["validation"]["base"],
            None if arm == "raw_prior" else development["validation"][arm],
        )
        heads, fit_records = diagnostic_head_path(
            development["train"],
            y_train,
            development["validation"],
            y_validation,
            arm,
            source_config,
            guard,
        )
        for head, fit in zip(heads, fit_records, strict=True):
            penalty = head["penalty"]
            scores = predict_head(
                head,
                development["validation"]["base"],
                None if arm == "raw_prior" else development["validation"][arm],
            )
            np.savez_compressed(output / f"candidate_{arm}_{penalty:g}.npz", **head)
            tail = validation_tail_sensitivity(
                scores,
                truth["validation"],
                observed["validation"].randomizers,
                seed=config["seed"],
                partitions=config["selection"]["calibration_partitions"],
                rates=config["selection"]["false_positive_rates"],
                strength=config["selection"]["primary_expected_signal"],
            )
            row = {**fit, "tail_partitions": tail}
            original = choices[arm]["candidates"][choices[arm]["selected_candidate"]]["penalty"]
            if penalty == original:
                np.testing.assert_allclose(scores, saved_scores, atol=1e-8, rtol=1e-8)
                row["saved_choice_score_replay_max_difference"] = float(
                    np.max(np.abs(scores - saved_scores))
                )
            candidates.append(row)
            LOGGER.info("Selection diagnosis %s lambda=%g complete", arm, penalty)
        ce_selected = int(np.argmin([row["validation_balanced_log_loss"] for row in candidates]))
        tail_selection = {}
        for rate in config["selection"]["false_positive_rates"]:
            values = np.asarray(
                [
                    [
                        part["primary_efficiency"]
                        for part in row["tail_partitions"]
                        if part["target_false_positive_rate"] == rate
                    ]
                    for row in candidates
                ]
            )
            eligible = np.asarray([row["passes_original_gradient_gate"] for row in candidates])
            ranked = np.where(eligible[:, None], values, -np.inf)
            best = int(np.argmax(ranked.mean(1)))
            winners = np.argmax(ranked, axis=0)
            tied = ranked == ranked.max(0)
            tail_selection[str(rate)] = {
                "tail_preferred_penalty": candidates[best]["penalty"],
                "log_loss_preferred_penalty": candidates[ce_selected]["penalty"],
                "efficiency_by_penalty_mean": values.mean(1).tolist(),
                "efficiency_by_penalty_min": values.min(1).tolist(),
                "efficiency_by_penalty_max": values.max(1).tolist(),
                "first_place_fraction": [
                    float((winners == index).mean()) for index in range(len(candidates))
                ],
                "tie_adjusted_first_place_fraction": (tied / tied.sum(0)).mean(1).tolist(),
                "tail_minus_log_loss_mean_efficiency": float(
                    (values[best] - values[ce_selected]).mean()
                ),
                "ranges_are_confidence_intervals": False,
                "candidate_passes_gradient_gate": eligible.tolist(),
            }
        result[arm] = {
            "candidates": candidates,
            "log_loss_selected_index": ce_selected,
            "tail_selection": tail_selection,
        }
    return result


def _timing_targets(labels, reference):
    centers = (reference["edges"][:-1] + reference["edges"][1:]) / 2
    by_source = []
    for source in labels["sources_m"]:
        marginal = analytic_location_pdf(
            reference["positions"], reference["edges"], source, 1.0, 0.0
        ).sum(0)
        cumulative = np.cumsum(marginal)
        cumulative[-1] = 1
        quantiles = centers[np.searchsorted(cumulative, [0.1, 0.5, 0.9])]
        by_source.append([marginal @ centers, *quantiles])
    selected = labels["expected_signal"] > 0
    return selected, np.asarray(by_source)[labels["source_group"][selected]]


def _timing_design(values, selected, method, pdf_indices, hidden_indices):
    base = values["base"][selected]
    if method == "raw_prior":
        return base, np.zeros((len(base), values["self_split"].shape[1]), dtype=np.float32)
    arm = "self_split" if method.startswith("self_") else "untrained"
    extra = values[arm][selected].copy()
    if method.endswith("_full"):
        return base, extra
    keep = pdf_indices if method.endswith("_pdf_only") else hidden_indices
    removed = np.ones(extra.shape[1], dtype=bool)
    removed[keep] = False
    extra[:, removed] = 0
    return np.zeros_like(base), extra


def _timing_summary(prediction, target, labels, counts, selected):
    errors = np.abs(prediction - target)
    strengths, counts = labels["expected_signal"][selected], counts[selected]
    rows = {
        "all": np.ones(len(target), dtype=bool),
        "recorded_empty": counts == 0,
        "recorded_1_to_5": (counts >= 1) & (counts <= 5),
        "recorded_above_5": counts > 5,
    }
    rows.update(
        {f"expected_signal_{level:g}": strengths == level for level in np.unique(strengths)}
    )
    return {
        label: {
            "windows": int(mask.sum()),
            "mae_ns": errors[mask].mean(0).tolist() if mask.any() else None,
        }
        for label, mask in rows.items()
    }


def timing_diagnosis(config, source_config, development, truth, observed, reference, output, guard):
    """Use injected timing labels only in fixed auxiliary training/validation decoders."""
    train_selected, y_train = _timing_targets(truth["train"], reference)
    validation_selected, y_validation = _timing_targets(truth["validation"], reference)
    settings = source_config["features"]
    # Three encoders each have mean/max of a 32-channel sensor context.
    hidden = 2 * 32 * len(settings["random_encoder_seeds"])
    pdf_indices, hidden_indices = pdf_hidden_indices(
        development["train"]["self_split"].shape[1],
        settings["sensor_groups"],
        settings["time_groups"],
        hidden,
    )
    counts = np.diff(observed["validation"].offsets)
    constant = np.broadcast_to(y_train.mean(0), y_validation.shape)
    result = {
        "constant_training_mean": [
            {
                "penalty": None,
                "strata": _timing_summary(
                    constant, y_validation, truth["validation"], counts, validation_selected
                ),
            }
        ]
    }
    methods = (
        "raw_prior",
        "self_full",
        "untrained_full",
        "self_pdf_only",
        "untrained_pdf_only",
        "self_hidden_only",
        "untrained_hidden_only",
    )
    for method in methods:
        base, extra = _timing_design(
            development["train"], train_selected, method, pdf_indices, hidden_indices
        )
        heads = ridge_timing_path(
            base, extra, y_train, source_config["head"], config["timing"]["ridge_penalties"]
        )
        validation_base, validation_extra = _timing_design(
            development["validation"], validation_selected, method, pdf_indices, hidden_indices
        )
        result[method] = []
        for head in heads:
            prediction = predict_timing(head, validation_base, validation_extra)
            np.savez_compressed(output / f"timing_{method}_{head['penalty']:g}.npz", **head)
            result[method].append(
                {
                    "penalty": head["penalty"],
                    "strata": _timing_summary(
                        prediction, y_validation, truth["validation"], counts, validation_selected
                    ),
                }
            )
            guard()
        LOGGER.info("Timing diagnosis %s complete", method)
    return result


def _extractor_from_saved_states(store, source_config, reference, device):
    models = []
    seeds = source_config["features"]["random_encoder_seeds"]
    for seed in seeds:
        state = torch.load(
            store.path(f"frozen_seed_{seed}.pt"), map_location="cpu", weights_only=True
        )["state_dict"]
        model = DensePhotonReference(state["temporal.0.weight"].shape[0]).to(device).eval()
        model.load_state_dict(state)
        models.append(model)
    scorer = SimpleNamespace(
        models=models,
        shape=reference["prior"].shape,
        device=torch.device(device),
        coordinates=torch.as_tensor(reference["positions"], dtype=torch.float32, device=device),
        live=torch.ones(reference["prior"].shape[0], dtype=torch.bool, device=device),
    )
    extractor = FrozenFeatureExtractor(scorer, source_config["features"])
    for seed, model in zip(seeds, extractor.models["untrained"], strict=True):
        model.load_state_dict(
            torch.load(store.path(f"random_seed_{seed}.pt"), map_location="cpu", weights_only=True)[
                "state_dict"
            ]
        )
    extractor.empty = {
        arm: extractor._forward(np.zeros((1, *scorer.shape), dtype=np.int64), arm)
        for arm in extractor.models
    }
    return extractor


def _score_parts(head, base, extra):
    values = feature_design(base, extra, head["extra_dimension"]).astype(np.float64)
    values = np.clip((values - head["mean"]) / head["scale"], -head["clip"], head["clip"])
    split = base.shape[1]
    return values[:, :split] @ head["weight"][:split], values[:, split:] @ head["weight"][split:]


def _summarize_cases(metrics, masks):
    return {
        name: {
            "windows": int(mask.sum()),
            **{
                key: {
                    "mean": float(value[mask].mean()),
                    "median": float(np.median(value[mask])),
                    "p90": float(np.quantile(value[mask], 0.9)),
                }
                if mask.any()
                else None
                for key, value in metrics.items()
            },
        }
        for name, mask in masks.items()
    }


@torch.inference_mode()
def coupling_diagnosis(
    store, config, source_config, development, truth, observed, reference, output, device, guard
):
    """Break pairings with fixed marginals and measure only development responses."""
    count = min(config["coupling"]["validation_windows"], len(observed["validation"]))
    selected = np.sort(
        np.random.default_rng(config["seed"] + 1).choice(
            len(observed["validation"]), count, replace=False
        )
    )
    original = _subset(observed["validation"], selected)
    bins = reference["prior"].shape[1]
    permuted, flags = permute_sensor_time_pairings(original, bins, config["seed"] + 2)
    extractor = _extractor_from_saved_states(store, source_config, reference, device)
    heads = {arm: _head(store, arm) for arm in HEAD_ARMS}
    replay = {"base": 0.0, "self_split": 0.0, "untrained": 0.0}
    metrics = {
        arm: {
            key: np.empty(count)
            for key in (
                "pdf_tv",
                "time_marginal_tv",
                "sensor_marginal_tv",
                "hidden_relative_l2",
                "absolute_raw_logit_change",
                "absolute_extra_logit_change",
                "signed_total_logit_change",
            )
        }
        for arm in ("self_split", "untrained")
    }
    batch_size = config["runtime"]["batch_size"]
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        a, b = original.dense_batch(start, stop, reference["prior"].shape)
        pa, pb = permuted.dense_batch(start, stop, reference["prior"].shape)
        np.testing.assert_array_equal(a.sum(1), pa.sum(1))
        np.testing.assert_array_equal(a.sum(2), pa.sum(2))
        np.testing.assert_array_equal(b.sum(1), pb.sum(1))
        np.testing.assert_array_equal(b.sum(2), pb.sum(2))
        base, _ = observation_features(original, start, stop, a, b, reference)
        base_p, _ = observation_features(permuted, start, stop, pa, pb, reference)
        np.testing.assert_allclose(
            base, development["validation"]["base"][selected[start:stop]], atol=1e-6, rtol=1e-6
        )
        replay["base"] = max(
            replay["base"],
            float(np.max(np.abs(base - development["validation"]["base"][selected[start:stop]]))),
        )
        for arm in ("self_split", "untrained"):
            qa, ha = extractor.predict_tensors(a, arm)
            qb, hb = extractor.predict_tensors(b, arm)
            pqa, pha = extractor.predict_tensors(pa, arm)
            pqb, phb = extractor.predict_tensors(pb, arm)
            extra, _ = encoder_features_tensor(
                a, b, qa, qb, ha, hb, reference["prior"], source_config["features"]
            )
            extra_p, _ = encoder_features_tensor(
                pa, pb, pqa, pqb, pha, phb, reference["prior"], source_config["features"]
            )
            np.testing.assert_allclose(
                extra, development["validation"][arm][selected[start:stop]], atol=3e-5, rtol=1e-5
            )
            replay[arm] = max(
                replay[arm],
                float(np.max(np.abs(extra - development["validation"][arm][selected[start:stop]]))),
            )
            q, pq = 0.5 * (qa + qb), 0.5 * (pqa + pqb)
            h, ph = torch.cat([ha, hb], 1), torch.cat([pha, phb], 1)
            row = metrics[arm]
            row["pdf_tv"][start:stop] = (0.5 * (pq - q).abs().sum((1, 2))).cpu().numpy()
            row["time_marginal_tv"][start:stop] = (
                (0.5 * (pq.sum(1) - q.sum(1)).abs().sum(1)).cpu().numpy()
            )
            row["sensor_marginal_tv"][start:stop] = (
                (0.5 * (pq.sum(2) - q.sum(2)).abs().sum(1)).cpu().numpy()
            )
            row["hidden_relative_l2"][start:stop] = (
                ((ph - h).norm(dim=1) / h.norm(dim=1).clamp_min(1e-12)).cpu().numpy()
            )
            raw, added = _score_parts(heads[arm], base, extra)
            raw_p, added_p = _score_parts(heads[arm], base_p, extra_p)
            row["absolute_raw_logit_change"][start:stop] = np.abs(raw_p - raw)
            row["absolute_extra_logit_change"][start:stop] = np.abs(added_p - added)
            row["signed_total_logit_change"][start:stop] = raw_p + added_p - raw - added
        guard()
    labels = truth["validation"]["expected_signal"][selected]
    masks = {
        "all": np.ones(count, dtype=bool),
        "changed": flags["half_joint_changed"],
        "unchanged": ~flags["half_joint_changed"],
        "changed_signal": flags["half_joint_changed"] & (labels > 0),
        "changed_background": flags["half_joint_changed"] & (labels == 0),
    }
    saved = {"validation_row_id": selected, "expected_signal": labels, **flags}
    for arm, values in metrics.items():
        saved.update({f"{arm}_{name}": value for name, value in values.items()})
    np.savez_compressed(output / "coupling_cases.npz", **saved)
    return {
        "selected_windows": count,
        "eligible_windows": int(flags["eligible"].sum()),
        "changed_windows": int(flags["half_joint_changed"].sum()),
        "full_joint_changed_windows": int(flags["full_joint_changed"].sum()),
        "per_half_marginals_verified_exactly": True,
        "cached_feature_replay_max_difference": replay,
        "methods": {arm: _summarize_cases(values, masks) for arm, values in metrics.items()},
    }


def run_diagnostics(config, device):
    """Run a bounded diagnosis behind the development-only data accessor."""
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    store = DevelopmentStore(config["source"])
    source_config = store.json("config.json")
    head_lock = store.json("head_lock.json")
    if (
        hashlib.sha256(json.dumps(source_config, sort_keys=True).encode()).hexdigest()
        != head_lock["identity"]["config_sha256"]
    ):
        message = "The source development configuration changed after its original head lock"
        raise ValueError(message)
    write_json(output / "config.json", config)
    shutil.copyfile(config["protocol"], output / "protocol.txt")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started, peaks = time.perf_counter(), {"process_rss_gib": 0.0, "gpu_allocated_gib": 0.0}

    def guard():
        _resource_guard(config, started, torch.device(device), peaks)

    code = (
        "src/tools/neutrino_development.py",
        "src/tools/neutrino_development_visualize.py",
        "src/ssunet/neutrino_development.py",
        "src/ssunet/neutrino_detection_probe.py",
        "src/ssunet/models/neutrino_dense.py",
        "src/tools/neutrino_detection_probe.py",
        "src/tools/neutrino_analytic_study.py",
    )
    identity = {
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "protocol_sha256": shard_sha256(config["protocol"]),
        "code_sha256": {name: shard_sha256(name) for name in code},
        "source_head_lock_sha256": store.digest("head_lock.json"),
    }
    write_json(output / "identity.json", identity)
    for name in code:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    development, truth, observed = {}, {}, {}
    for fold in ("train", "validation"):
        development[fold] = store.arrays(f"{fold}/features.npz")
        truth[fold] = store.arrays(f"{fold}/injection_labels.npz")
        observed[fold] = PhotonWindows(**store.arrays(f"{fold}/observations.npz"))
    reference = store.arrays("frozen_reference.npz")
    reference.update(
        {
            "background": source_config["generator"]["background_expected_photons"],
            "amplitudes": source_config["generator"]["signal_expected_photons"],
            "mixture_fraction": config["coupling"]["source_reference_mixture_fraction"],
        }
    )
    selection = selection_diagnosis(
        store, config, source_config, development, truth, observed, output, guard
    )
    write_json(output / "selection.json", selection)
    timing = timing_diagnosis(
        config, source_config, development, truth, observed, reference, output, guard
    )
    write_json(output / "timing.json", timing)
    coupling = coupling_diagnosis(
        store, config, source_config, development, truth, observed, reference, output, device, guard
    )
    write_json(output / "coupling.json", coupling)
    digests = {name: store.digest(name) for name in sorted(store.accessed)}
    for name, digest in digests.items():
        if name in head_lock["artifact_sha256"] and digest != head_lock["artifact_sha256"][name]:
            message = f"Development input changed since the original probe: {name}"
            raise ValueError(message)
    for name, digest in identity["code_sha256"].items():
        if shard_sha256(name) != digest:
            message = "Diagnostic code changed during execution"
            raise RuntimeError(message)
    write_json(
        output / "data_access.json",
        {
            "accessed_files": sorted(store.accessed),
            "input_sha256": digests,
            "test_or_calibration_data_read": False,
        },
    )
    result = {
        "scope": "post_hoc_development_diagnosis_train_validation_only",
        "selection": selection,
        "timing": timing,
        "coupling": coupling,
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory": peaks,
        "source_validation_sources": len(truth["validation"]["sources_m"]),
        "test_or_calibration_data_read": False,
    }
    write_json(output / "results.json", result)
    create_development_report(output, config, result)
    LOGGER.info("Completed development-only diagnosis: %s", output / "index.html")
    return result


def main():
    """Run once into a new directory, without opening any calibration/test outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/neutrino_development_diagnostics.yml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_diagnostics(yaml.safe_load(Path(args.config).read_text()), args.device)


if __name__ == "__main__":
    main()
