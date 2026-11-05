"""Fit matched event heads on fresh observations using frozen learned/random encoders."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..ssunet.datasets.neutrino import shard_sha256
from ..ssunet.models.neutrino_dense import DensePhotonReference
from ..ssunet.neutrino_detection import (
    calibrated_tail_probabilities,
    sample_windows,
    summarize_detection,
)
from ..ssunet.neutrino_detection_probe import (
    BASE_SCALARS,
    HEAD_ARMS,
    encoder_features_tensor,
    fit_head_candidates,
    observation_features,
    predict_head,
)
from ..ssunet.photon_reference import photon_count_log_probabilities
from .neutrino_detection import FrozenScorer, _resource_guard, _write_table
from .neutrino_detection_probe_visualize import create_probe_report
from .neutrino_prepare import write_json

LOGGER = logging.getLogger(__name__)


class FrozenFeatureExtractor:
    """Read PDF and hidden features from frozen copies of the same dense backbone."""

    def __init__(self, scorer, settings):
        """Preserve selected states and construct reproducible untrained controls."""
        self.scorer = scorer
        self.models = {"self_split": scorer.models, "untrained": []}
        channels = scorer.models[0].temporal[0].out_channels
        for seed in settings["random_encoder_seeds"]:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = DensePhotonReference(channels).to(scorer.device).eval()
            self.models["untrained"].append(model)
        for models in self.models.values():
            for model in models:
                model.requires_grad_(False)
        self.empty = {
            arm: self._forward(np.zeros((1, *scorer.shape), dtype=np.int64), arm)
            for arm in self.models
        }

    @torch.inference_mode()
    def _forward(self, counts, arm):
        scorer = self.scorer
        observed = torch.as_tensor(counts, device=scorer.device)
        p = torch.full((len(observed),), 0.5, device=scorer.device)
        probability = torch.zeros(observed.shape, dtype=torch.float64, device=scorer.device)
        hidden = []
        for model in self.models[arm]:
            captured = []

            def capture_context(module, inputs, output, captured=captured):
                del module, inputs
                captured.append(output)

            handle = model.sensor_context.register_forward_hook(capture_context)
            try:
                logits = model(observed, scorer.coordinates, scorer.live, p)
            finally:
                handle.remove()
            probability += photon_count_log_probabilities(logits, scorer.live).double().exp()
            context = captured[0]
            hidden.append(torch.cat([context.mean(1), context.amax(1)], dim=1))
        probability /= len(self.models[arm])
        probability /= probability.sum((1, 2), keepdim=True)
        return probability, torch.cat(hidden, dim=1)

    @torch.inference_mode()
    def predict_tensors(self, counts, arm):
        """Retain full PDFs on device and reuse the deterministic empty result."""
        active = counts.sum((1, 2)) > 0
        empty_pdf, empty_hidden = self.empty[arm]
        pdf = empty_pdf.expand(counts.shape).clone()
        hidden = empty_hidden.expand(len(counts), empty_hidden.shape[1]).clone()
        if active.any():
            mask = torch.as_tensor(active, device=self.scorer.device)
            pdf[mask], hidden[mask] = self._forward(counts[active], arm)
        return pdf, hidden

    def predict(self, counts, arm):
        """Expose arrays for diagnostics while the main path transfers only summaries."""
        pdf, hidden = self.predict_tensors(counts, arm)
        return pdf.cpu().numpy(), hidden.cpu().numpy()


def _reference(scorer, config):
    return {
        "prior": scorer.prior,
        "positions": scorer.positions,
        "sensor_kernel": scorer.sensor_kernel,
        "time_kernel": scorer.time_kernel,
        "mixture_fraction": scorer.mixture["prior_fraction"],
        "background": config["generator"]["background_expected_photons"],
        "amplitudes": scorer.config["scoring"]["template_signal_grid"],
    }


def extract_features(windows, extractor, config, fold, guard):
    """Make compact observation-only features once for all heads and fixed controls."""
    scorer = extractor.scorer
    reference = _reference(scorer, config)
    features, controls = {}, {}
    batch_size = config["runtime"]["batch_size"]
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        a, b = windows.dense_batch(start, stop, scorer.shape)
        base, batch_controls = observation_features(windows, start, stop, a, b, reference)
        batch_features = {"base": base}
        batch_controls["physics_template"] = scorer.template_scores(windows, start, stop)
        for arm in ("self_split", "untrained"):
            qa, ha = extractor.predict_tensors(a, arm)
            qb, hb = extractor.predict_tensors(b, arm)
            batch_features[arm], cross = encoder_features_tensor(
                a, b, qa, qb, ha, hb, scorer.prior, config["features"]
            )
            if arm == "self_split":
                batch_controls["frozen_score"] = batch_controls["count"] + cross
        if not features:
            features = {
                name: np.empty((len(windows), value.shape[1]), dtype=np.float32)
                for name, value in batch_features.items()
            }
            controls = {name: np.empty(len(windows)) for name in batch_controls}
        for name, values in batch_features.items():
            features[name][start:stop] = values
        for name, values in batch_controls.items():
            controls[name][start:stop] = values
        guard()
        if start // batch_size % 100 == 0:
            LOGGER.info("%s: extracted %d/%d windows", fold, stop, len(windows))
    return features, controls


def generate_fold(config, fold, scorer, streams, excluded):
    """Generate independent source groups; return truth separately from observations."""
    settings, partition = config["generator"], config["generator"][fold]
    groups, background = partition["source_groups"], partition["background_windows"]
    levels = np.asarray(settings["signal_expected_photons"], dtype=float)
    sources = np.random.default_rng(streams[0]).uniform(
        settings["source_min_m"], settings["source_max_m"], size=(groups, 3)
    )
    if len(sources) and len(excluded) and (sources[:, None] == excluded[None]).all(2).any():
        message = "New source positions overlap a previous study or another probe fold"
        raise ValueError(message)
    per_source = len(levels) * partition["repetitions_per_source"]
    strength = np.concatenate(
        [
            np.zeros(background),
            np.tile(np.repeat(levels, partition["repetitions_per_source"]), groups),
        ]
    )
    group = np.concatenate([np.full(background, -1), np.repeat(np.arange(groups), per_source)])

    def pdfs():
        for _ in range(background):
            yield None
        for source in sources:
            pdf = scorer.signal_pdf(source)
            for _ in range(per_source):
                yield pdf

    observed = sample_windows(
        pdfs(),
        strength,
        settings["background_expected_photons"],
        scorer.prior.size,
        np.random.default_rng(streams[1]),
        p=settings["split_probability"],
    )
    return observed, {"expected_signal": strength, "source_group": group, "sources_m": sources}


def _old_sources(scorer, frozen_config):
    with np.load(
        scorer.prior_root / "observations.evaluation_only.npz", allow_pickle=False
    ) as data:
        original = data["latent_source_position_m"].copy()
    path = Path(frozen_config["output"]) / "evaluation_only_truth.npz"
    with np.load(path, allow_pickle=False) as data:
        previous_detection = data["sources_m"].copy()
    return np.concatenate([original, previous_detection])


def _save_fold(output, fold, observations, truth, features, controls):
    folder = output / fold
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / "observations.npz", **vars(observations))
    np.savez_compressed(folder / "injection_labels.npz", **truth)
    np.savez_compressed(folder / "features.npz", **features)
    np.savez_compressed(folder / "fixed_scores.npz", **controls)


def _head_scores(heads, features, config):
    return {
        arm: predict_head(
            head,
            features["base"],
            None if arm == "raw_prior" else features[arm],
            batch_size=config["runtime"]["head_batch_size"],
        )
        for arm, head in heads.items()
    }


def verify_probe_inputs(output, identity):
    """Reject altered selected heads, development evidence or background calibration."""
    head_lock = json.loads((output / "head_lock.json").read_text())
    if head_lock["identity"] != identity:
        message = "Frozen probe identity changed after head selection"
        raise ValueError(message)
    for name, digest in head_lock["artifact_sha256"].items():
        if shard_sha256(output / name) != digest:
            message = f"A locked head/development artifact changed: {name}"
            raise ValueError(message)
    if (output / "analysis_lock.json").exists():
        calibration = json.loads((output / "analysis_lock.json").read_text())
        if shard_sha256(output / "head_lock.json") != calibration["head_lock_sha256"]:
            message = "Head selection changed after background calibration"
            raise ValueError(message)
        for name, digest in calibration["artifact_sha256"].items():
            if shard_sha256(output / name) != digest:
                message = f"A locked calibration artifact changed: {name}"
                raise ValueError(message)


def _identity(config, scorer):
    files = (
        "src/tools/neutrino_detection_probe.py",
        "src/tools/neutrino_detection_probe_visualize.py",
        "src/ssunet/neutrino_detection_probe.py",
        "src/ssunet/neutrino_detection.py",
        "src/tools/neutrino_detection.py",
        "src/ssunet/models/neutrino_dense.py",
        "src/ssunet/photon_reference.py",
        "src/ssunet/distributions.py",
        "src/tools/neutrino_analytic_study.py",
        "src/ssunet/neutrino_streaming.py",
        "src/ssunet/datasets/neutrino.py",
        "src/tools/neutrino_prepare.py",
    )
    frozen_config = yaml.safe_load(Path(config["frozen_config"]).read_text())
    return {
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "protocol_sha256": shard_sha256(config["protocol"]),
        "frozen_config_sha256": shard_sha256(config["frozen_config"]),
        "code_sha256": {name: shard_sha256(name) for name in files},
        "checkpoints": scorer.checkpoints,
        "earlier_detection_lock_sha256": shard_sha256(
            Path(frozen_config["output"]) / "analysis_lock.json"
        ),
        "earlier_detection_sources_sha256": shard_sha256(
            Path(frozen_config["output"]) / "evaluation_only_truth.npz"
        ),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def validate_probe_config(config, frozen_config):
    """Reject unsupported source/exposure contracts before opening an output directory."""
    generator = config["generator"]
    if (
        generator["split_probability"] != 0.5
        or generator["background_expected_photons"]
        != frozen_config["generator"]["background_expected_photons"]
    ):
        message = "This frozen probe requires the original split/background exposure"
        raise ValueError(message)
    levels = np.asarray(generator["signal_expected_photons"])
    if np.any(levels <= 0) or not np.isfinite(levels).all() or not np.all(np.diff(levels) > 0):
        message = "Use ordered unique positive expected signal strengths"
        raise ValueError(message)
    for fold in ("train", "validation", "test"):
        partition = generator[fold]
        if (
            partition["source_groups"] < 2
            or min(partition["repetitions_per_source"], partition["background_windows"]) < 1
        ):
            message = "Each labelled fold requires independent sources, repeats and background"
            raise ValueError(message)
    if (
        generator["calibration"]["source_groups"] != 0
        or generator["calibration"]["background_windows"] < 1
    ):
        message = "Calibration requires a separate background-only cohort"
        raise ValueError(message)
    if len(config["features"]["random_encoder_seeds"]) != len(
        frozen_config["scoring"]["model_seeds"]
    ):
        message = "Learned and untrained ensembles must have the same parameter budget"
        raise ValueError(message)
    if config["head"]["selection_metric"] != "balanced_validation_log_loss":
        message = "Only the registered balanced validation selection rule is implemented"
        raise ValueError(message)


def run_probe(config, device):
    """Fit on new folds and lock heads plus calibration before a single fresh test."""
    frozen_config = yaml.safe_load(Path(config["frozen_config"]).read_text())
    validate_probe_config(config, frozen_config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    shutil.copyfile(config["protocol"], output / "protocol.txt")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    peaks = {"process_rss_gib": 0.0, "gpu_allocated_gib": 0.0}
    scorer = FrozenScorer(frozen_config, device)
    extractor = FrozenFeatureExtractor(scorer, config["features"])
    identity = _identity(config, scorer)

    def guard():
        _resource_guard(config, started, scorer.device, peaks)

    for name in identity["code_sha256"]:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    for seed, item in scorer.checkpoints.items():
        shutil.copyfile(item["path"], output / f"frozen_seed_{seed}.pt")
    for seed, model in zip(
        config["features"]["random_encoder_seeds"], extractor.models["untrained"], strict=True
    ):
        torch.save(
            {"state_dict": model.state_dict(), "seed": seed}, output / f"random_seed_{seed}.pt"
        )
    np.savez_compressed(
        output / "frozen_reference.npz",
        prior=scorer.prior,
        positions=scorer.positions,
        edges=scorer.edges,
        sensor_kernel=scorer.sensor_kernel,
        time_kernel=scorer.time_kernel,
        template_sources=scorer.template_sources,
    )
    streams = np.random.SeedSequence(config["generator"]["seed"]).spawn(8)
    excluded = _old_sources(scorer, frozen_config)
    development, labels = {}, {}
    for index, fold in enumerate(("train", "validation")):
        observed, truth = generate_fold(
            config, fold, scorer, streams[index * 2 : index * 2 + 2], excluded
        )
        excluded = np.concatenate([excluded, truth["sources_m"]])
        features, controls = extract_features(observed, extractor, config, fold, guard)
        _save_fold(output, fold, observed, truth, features, controls)
        development[fold] = features
        labels[fold] = (truth["expected_signal"] > 0).astype(np.float64)
        del observed, truth, features, controls
    dimensions = {name: values.shape[1] for name, values in development["train"].items()}
    write_json(
        output / "feature_contract.json",
        {
            "dimensions": dimensions,
            "base_scalar_order": BASE_SCALARS,
            "labels_enter_features": False,
        },
    )
    heads, selections = {}, {}
    for arm in HEAD_ARMS:
        LOGGER.info("Fitting fixed logistic penalty grid: %s", arm)
        heads[arm], selections[arm] = fit_head_candidates(
            development["train"],
            labels["train"],
            development["validation"],
            labels["validation"],
            arm,
            config,
            guard,
        )
        np.savez_compressed(output / f"head_{arm}.npz", **heads[arm])
        write_json(output / f"selection_{arm}.json", selections[arm])
        LOGGER.info("Selected %s penalty=%g", arm, heads[arm]["penalty"])
    write_json(output / "head_selection.json", selections)
    del development, labels
    artifacts = [
        *[f"head_{arm}.npz" for arm in HEAD_ARMS],
        "head_selection.json",
        "feature_contract.json",
        "frozen_reference.npz",
        *[f"random_seed_{seed}.pt" for seed in config["features"]["random_encoder_seeds"]],
        *[f"frozen_seed_{seed}.pt" for seed in scorer.checkpoints],
        *[
            f"{fold}/{name}.npz"
            for fold in ("train", "validation")
            for name in ("features", "observations", "injection_labels", "fixed_scores")
        ],
    ]
    head_lock = {
        "identity": identity,
        "artifact_sha256": {name: shard_sha256(output / name) for name in artifacts},
    }
    write_json(output / "head_lock.json", head_lock)
    verify_probe_inputs(output, identity)
    observed, truth = generate_fold(config, "calibration", scorer, streams[4:6], excluded)
    features, controls = extract_features(observed, extractor, config, "calibration", guard)
    calibration_scores = {**controls, **_head_scores(heads, features, config)}
    _save_fold(output, "calibration", observed, truth, features, controls)
    np.savez_compressed(output / "calibration_scores.npz", **calibration_scores)
    del observed, truth, features, controls
    if _identity(config, scorer) != identity:
        message = "Frozen code or encoder inputs changed before the probe test"
        raise RuntimeError(message)
    analysis_lock = {
        "head_lock_sha256": shard_sha256(output / "head_lock.json"),
        "artifact_sha256": {
            name: shard_sha256(output / name)
            for name in (
                "calibration_scores.npz",
                "calibration/observations.npz",
                "calibration/features.npz",
                "calibration/injection_labels.npz",
                "calibration/fixed_scores.npz",
            )
        },
    }
    write_json(output / "analysis_lock.json", analysis_lock)
    verify_probe_inputs(output, identity)
    write_json(
        output / "test_opened.json",
        {"analysis_lock_sha256": shard_sha256(output / "analysis_lock.json")},
    )
    observed, truth = generate_fold(config, "test", scorer, streams[6:8], excluded)
    features, controls = extract_features(observed, extractor, config, "test", guard)
    scores = {**controls, **_head_scores(heads, features, config)}
    _save_fold(output, "test", observed, truth, features, controls)
    np.savez_compressed(output / "test_scores.npz", **scores)
    probabilities = {
        name: calibrated_tail_probabilities(calibration_scores[name], value, observed.randomizers)
        for name, value in scores.items()
    }
    np.savez_compressed(output / "test_tail_probabilities.npz", **probabilities)
    if _identity(config, scorer) != identity:
        message = "Frozen code or encoder inputs changed during the probe test"
        raise RuntimeError(message)
    verify_probe_inputs(output, identity)
    result = {
        "scope": "exploratory_labelled_head_on_frozen_encoder_exact_poisson_toy",
        "identity": identity,
        "test_windows": len(observed),
        "test_sources": config["generator"]["test"]["source_groups"],
        "source_overlap_with_prior_studies_or_other_folds": 0,
        "empty_test_windows": int((np.diff(observed.offsets) == 0).sum()),
        "runtime_seconds_through_test_extraction": time.perf_counter() - started,
        "peak_memory": peaks,
        "feature_dimensions": dimensions,
        "head_parameters_each": dimensions["base"] + dimensions["self_split"] + 1,
        "selected_penalties": {arm: head["penalty"] for arm, head in heads.items()},
        "operating_points": summarize_detection(probabilities, truth, config),
        "uncertainty": (
            "conditional source-group bootstrap; excludes calibration, encoder/head fitting "
            "and selection uncertainty"
        ),
        "self_splitting_objective_isolated": False,
        "real_event_recovery_validated": False,
        "encoder_parameters_fitted": 0,
        "head_uses_event_labels": True,
    }
    write_json(output / "test_results.json", result)
    _write_table(output / "detection_results.csv", result["operating_points"])
    create_probe_report(output, config, result, selections, probabilities, truth)
    LOGGER.info("Completed frozen-feature probe: %s", output / "index.html")
    return result


def main():
    """Run a fixed probe once, preserving every existing study directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/neutrino_detection_probe.yml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = yaml.safe_load(Path(args.config).read_text())
    run_probe(config, args.device)


if __name__ == "__main__":
    main()
