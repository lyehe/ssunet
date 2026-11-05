"""Run a frozen-model, synthetic subthreshold injection/recovery feasibility study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import psutil
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ..ssunet.datasets.neutrino import shard_sha256
from ..ssunet.models.neutrino_dense import DensePhotonReference
from ..ssunet.neutrino_detection import (
    calibrated_tail_probabilities,
    count_glr,
    cross_shape_score,
    kernel_at_photons,
    normal_kernel,
    sample_windows,
    summarize_detection,
)
from ..ssunet.neutrino_streaming import collect_training_reference
from ..ssunet.photon_reference import photon_count_log_probabilities
from .neutrino_analytic_study import _baseline_bank, analytic_location_pdf, open_analytic
from .neutrino_prepare import write_json

LOGGER = logging.getLogger(__name__)
LABELS = {
    "count": "Count only",
    "static_prior": "Static training prior",
    "geometry_kernel": "Geometry/time kernel",
    "physics_template": "Physics-template search",
    "self_split": "Frozen self-split ensemble",
}
COLORS = ["#667788", "#8c65b5", "#188396", "#2f935c", "#cc6b29"]


class FrozenScorer:
    """Expose only recorded counts and fixed geometry to selected frozen models."""

    def __init__(self, config, device):
        """Load existing checkpoints and a training-only reference, without refitting."""
        self.config, self.device = config, torch.device(device)
        prior_config = yaml.safe_load(Path(config["pretraining_config"]).read_text())
        self.prior_root = Path(prior_config["data"]["output"])
        self.lock = json.loads((self.prior_root / "analysis_lock.json").read_text())
        if (
            shard_sha256(self.prior_root / "observations.h5")
            != self.lock["identity"]["observations_sha256"]
        ):
            message = "The analytic pretraining observations changed"
            raise ValueError(message)
        recorded_code = {
            str(Path(name).resolve()): digest
            for name, digest in self.lock["identity"]["code_sha256"].items()
        }
        for name in (
            "src/ssunet/models/neutrino_dense.py",
            "src/ssunet/photon_reference.py",
            "src/ssunet/distributions.py",
        ):
            if shard_sha256(name) != recorded_code[str(Path(name).resolve())]:
                message = "The frozen analytic model implementation changed"
                raise ValueError(message)
        training = open_analytic(prior_config, "train")
        if not training.live.all():
            message = "This finite toy benchmark requires the declared fully live support"
            raise ValueError(message)
        self.positions, self.edges = training.positions_m.copy(), training.edges_ns.copy()
        reference = collect_training_reference(training)
        selection = json.loads((self.prior_root / "baseline_selection.json").read_text())
        if (
            shard_sha256(self.prior_root / "baseline_selection.json")
            != self.lock["baseline_selection_sha256"]
        ):
            message = "Frozen analytic baseline selection changed"
            raise ValueError(message)
        bank = _baseline_bank(prior_config, training, reference, "cpu", selection)
        self.prior = bank.priors["selected_prior"].double().numpy()
        self.prior /= self.prior.sum()
        self.mixture = selection["selected_mixture"]
        self.sensor_kernel = normal_kernel(self.positions, self.mixture["space_m"])
        times = ((self.edges[:-1] + self.edges[1:]) / 2)[:, None]
        self.time_kernel = normal_kernel(times, self.mixture["time_ns"])
        training.close()
        self.shape = self.prior.shape
        self.live = torch.ones(self.shape[0], dtype=torch.bool, device=self.device)
        self.coordinates = torch.as_tensor(self.positions, dtype=torch.float32, device=self.device)
        self.models, self.checkpoints = [], {}
        for seed in config["scoring"]["model_seeds"]:
            item = self.lock["checkpoints"][str(seed)]
            if shard_sha256(item["path"]) != item["sha256"]:
                message = "A frozen analytic checkpoint changed"
                raise ValueError(message)
            model = DensePhotonReference(prior_config["model"]["channels"]).to(self.device)
            state = torch.load(item["path"], map_location="cpu", weights_only=True)
            model.load_state_dict(state["state_dict"])
            self.models.append(model.eval())
            self.checkpoints[str(seed)] = item
        self.empty_probability = self._predict(np.zeros((1, *self.shape), dtype=np.int64))[0]
        settings = config["generator"]
        axes = [
            np.linspace(low, high, config["scoring"]["template_points_per_axis"])
            for low, high in zip(settings["source_min_m"], settings["source_max_m"], strict=True)
        ]
        self.template_sources = np.asarray(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        templates = [self.signal_pdf(source).ravel() for source in self.template_sources]
        self.templates = torch.as_tensor(np.asarray(templates), device=self.device)
        self.amplitudes = torch.as_tensor(
            config["scoring"]["template_signal_grid"], device=self.device, dtype=torch.float64
        )

    def signal_pdf(self, source):
        """Use the established analytic response for generation or fixed templates."""
        return analytic_location_pdf(self.positions, self.edges, source, 1.0, 0.0)

    @torch.inference_mode()
    def _predict(self, counts):
        values = torch.as_tensor(counts, device=self.device)
        fraction = torch.full(
            (len(values),), self.config["generator"]["split_probability"], device=self.device
        )
        probability = torch.zeros(values.shape, dtype=torch.float64, device=self.device)
        for model in self.models:
            logits = model(values, self.coordinates, self.live, fraction)
            probability += photon_count_log_probabilities(logits, self.live).double().exp()
        probability /= len(self.models)
        probability /= probability.sum((1, 2), keepdim=True)
        return probability.cpu().numpy()

    def predict(self, counts):
        """Reuse the exact same deterministic empty prediction to avoid wasted work."""
        active = counts.sum((1, 2)) > 0
        probability = np.broadcast_to(self.empty_probability, counts.shape).copy()
        if active.any():
            probability[active] = self._predict(counts[active])
        return probability

    @torch.inference_mode()
    def template_scores(self, windows, start, stop):
        """Vectorize a fixed full template search using only observed photon locations."""
        lengths = np.diff(windows.offsets[start : stop + 1])
        maximum = int(lengths.max(initial=0))
        locations = np.zeros((stop - start, maximum), dtype=np.int64)
        mask = np.zeros_like(locations, dtype=bool)
        for row, index in enumerate(range(start, stop)):
            selected = windows.locations[windows.offsets[index] : windows.offsets[index + 1]]
            locations[row, : len(selected)], mask[row, : len(selected)] = selected, True
        ratio = self.templates[:, torch.as_tensor(locations, device=self.device)]
        ratio = ratio.permute(1, 0, 2) * self.prior.size
        ratio /= self.config["generator"]["background_expected_photons"]
        values = torch.log1p(self.amplitudes[None, :, None, None] * ratio[:, None])
        values *= torch.as_tensor(mask, device=self.device)[:, None, None, :]
        scores = values.sum(-1) - self.amplitudes[None, :, None]
        return scores.amax((1, 2)).clamp_min(0).cpu().numpy()


def _resource_guard(config, started, device, peaks):
    """Stop within fixed memory/time ceilings and retain measured peak usage."""
    memory = psutil.Process().memory_info()
    peaks["process_rss_gib"] = max(
        peaks["process_rss_gib"], getattr(memory, "peak_wset", memory.rss) / 2**30
    )
    if device.type == "cuda":
        peaks["gpu_allocated_gib"] = max(
            peaks["gpu_allocated_gib"], torch.cuda.max_memory_allocated(device) / 2**30
        )
    if (
        time.perf_counter() - started > config["runtime"]["max_seconds"]
        or peaks["process_rss_gib"] > config["runtime"]["max_process_rss_gib"]
        or peaks["gpu_allocated_gib"] > config["runtime"]["max_gpu_allocated_gib"]
    ):
        message = "Finite detection study resource ceiling reached; preserve partial artifacts"
        raise RuntimeError(message)


def score_windows(windows, scorer, started, peaks):
    """Compute every fixed statistic on the same photons, without injection truth."""
    config = scorer.config
    totals = np.diff(windows.offsets)
    base = count_glr(totals, config["generator"]["background_expected_photons"])
    result = {name: base.copy() for name in LABELS}
    batch_size = config["runtime"]["batch_size"]
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        a, b = windows.dense_batch(start, stop, scorer.shape)
        qa, qb = scorer.predict(a), scorer.predict(b)
        result["physics_template"][start:stop] = scorer.template_scores(windows, start, stop)
        for row, index in enumerate(range(start, stop)):
            left, right = windows.halves(index)
            prior_left, prior_right = scorer.prior.ravel()[left], scorer.prior.ravel()[right]
            result["static_prior"][index] += cross_shape_score(
                left, right, prior_right, prior_left, scorer.prior.size
            )
            kernel = [
                kernel_at_photons(
                    observed,
                    queried,
                    scorer.prior,
                    scorer.sensor_kernel,
                    scorer.time_kernel,
                    scorer.mixture["prior_fraction"],
                )
                for observed, queried in ((left, right), (right, left))
            ]
            result["geometry_kernel"][index] += cross_shape_score(
                left, right, *kernel, scorer.prior.size
            )
            result["self_split"][index] += cross_shape_score(
                left, right, qa[row].ravel()[right], qb[row].ravel()[left], scorer.prior.size
            )
        _resource_guard(config, started, scorer.device, peaks)
        if start // batch_size % 100 == 0:
            LOGGER.info(
                "Scored %d/%d windows (%.1fs)", stop, len(windows), time.perf_counter() - started
            )
    return result


def _save_windows(path, windows):
    np.savez_compressed(path, **vars(windows))


def _identity(config, scorer):
    code_paths = [
        Path(__file__),
        Path("src/ssunet/neutrino_detection.py"),
        Path("src/ssunet/models/neutrino_dense.py"),
        Path("src/ssunet/photon_reference.py"),
        Path("src/ssunet/distributions.py"),
        Path("src/tools/neutrino_analytic_study.py"),
    ]
    return {
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "pretraining_config_sha256": shard_sha256(config["pretraining_config"]),
        "protocol_sha256": shard_sha256(config["protocol"]),
        "code_sha256": {
            str(path.resolve().relative_to(Path.cwd())): shard_sha256(path) for path in code_paths
        },
        "checkpoints": scorer.checkpoints,
        "pretraining_lock_sha256": shard_sha256(scorer.prior_root / "analysis_lock.json"),
        "baseline_selection_sha256": shard_sha256(scorer.prior_root / "baseline_selection.json"),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def _test_observations(config, scorer, streams):
    settings = config["generator"]
    count = settings["test_background_windows"]
    groups = settings["test_source_groups"]
    levels = np.asarray(settings["signal_expected_photons"], dtype=float)
    sources = np.random.default_rng(streams[0]).uniform(
        settings["source_min_m"], settings["source_max_m"], size=(groups, 3)
    )
    with np.load(scorer.prior_root / "observations.evaluation_only.npz") as original:
        overlap = (sources[:, None] == original["latent_source_position_m"][None]).all(2).any()
    if overlap:
        message = "Fresh detection source positions overlap the original analytic study"
        raise ValueError(message)
    per_group = len(levels) * settings["repetitions_per_source"]
    group_ids = np.concatenate([np.full(count, -1), np.repeat(np.arange(groups), per_group)])
    signal = np.concatenate(
        [np.zeros(count), np.tile(np.repeat(levels, settings["repetitions_per_source"]), groups)]
    )

    def pdfs():
        for _ in range(count):
            yield None
        for source in sources:
            pdf = scorer.signal_pdf(source)
            for _ in range(per_group):
                yield pdf

    windows = sample_windows(
        pdfs(),
        signal,
        settings["background_expected_photons"],
        scorer.prior.size,
        np.random.default_rng(streams[1]),
        p=settings["split_probability"],
    )
    truth = {"expected_signal": signal, "source_group": group_ids, "sources_m": sources}
    return windows, truth


def validate_config(config):
    """Reject invalid or unsupported study contracts before creating any artifacts."""
    settings, evaluation = config["generator"], config["evaluation"]
    levels = np.asarray(settings["signal_expected_photons"], dtype=float)
    if (
        len(levels) < 3
        or np.any(levels <= 0)
        or not np.isfinite(levels).all()
        or not np.all(np.diff(levels) > 0)
        or evaluation["primary_expected_signal"] not in levels
    ):
        message = "Use ordered unique positive signal levels including the primary strength"
        raise ValueError(message)
    if settings["split_probability"] != 0.5:
        message = "Frozen analytic models require the registered split probability .5"
        raise ValueError(message)
    if settings["test_source_groups"] < 2 or settings["repetitions_per_source"] < 1:
        message = "At least two independent source groups and one repetition are required"
        raise ValueError(message)
    if min(settings["calibration_background_windows"], settings["test_background_windows"]) < 1:
        message = "Independent nonempty background calibration and test sets are required"
        raise ValueError(message)
    rates = evaluation["false_positive_rates"]
    if (
        not all(0 < rate < 1 for rate in rates)
        or evaluation["primary_false_positive_rate"] not in rates
    ):
        message = "Declare valid false-positive rates including the primary operating point"
        raise ValueError(message)


def run_detection(config, device):
    """Lock background calibration before generating and scoring a fresh test once."""
    validate_config(config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    shutil.copyfile(config["protocol"], output / "protocol.txt")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started, peaks = time.perf_counter(), {"process_rss_gib": 0, "gpu_allocated_gib": 0}
    scorer = FrozenScorer(config, device)
    identity = _identity(config, scorer)
    settings = config["generator"]
    streams = np.random.SeedSequence(settings["seed"]).spawn(3)
    calibration = sample_windows(
        (None for _ in range(settings["calibration_background_windows"])),
        np.zeros(settings["calibration_background_windows"]),
        settings["background_expected_photons"],
        scorer.prior.size,
        np.random.default_rng(streams[0]),
        p=settings["split_probability"],
    )
    _save_windows(output / "calibration_observations.npz", calibration)
    calibration_scores = score_windows(calibration, scorer, started, peaks)
    np.savez_compressed(output / "calibration_scores.npz", **calibration_scores)
    np.savez_compressed(
        output / "frozen_reference.npz",
        prior=scorer.prior,
        positions=scorer.positions,
        edges=scorer.edges,
        template_sources=scorer.template_sources,
        sensor_kernel=scorer.sensor_kernel,
        time_kernel=scorer.time_kernel,
    )
    for seed, item in scorer.checkpoints.items():
        shutil.copyfile(item["path"], output / f"frozen_seed_{seed}.pt")
    snapshots = output / "source_snapshot"
    for name in identity["code_sha256"]:
        target = snapshots / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    lock = {
        "identity": identity,
        "artifact_sha256": {
            name: shard_sha256(output / name)
            for name in (
                "calibration_scores.npz",
                "calibration_observations.npz",
                "frozen_reference.npz",
            )
        },
        "score_role": "calibrated_statistic_not_posterior_or_full_generative_likelihood_ratio",
        "scope": "exploratory_exact_poisson_toy_not_physical_neutrino_detection",
    }
    write_json(output / "analysis_lock.json", lock)
    write_json(
        output / "test_opened.json",
        {"analysis_lock_sha256": shard_sha256(output / "analysis_lock.json")},
    )
    windows, truth = _test_observations(config, scorer, streams[1:])
    _save_windows(output / "test_observations.npz", windows)
    np.savez_compressed(output / "evaluation_only_truth.npz", **truth)
    scores = score_windows(windows, scorer, started, peaks)
    np.savez_compressed(output / "test_scores.npz", **scores)
    probabilities = {
        name: calibrated_tail_probabilities(calibration_scores[name], value, windows.randomizers)
        for name, value in scores.items()
    }
    np.savez_compressed(output / "test_tail_probabilities.npz", **probabilities)
    if _identity(config, scorer) != identity:
        message = "Frozen implementation or inputs changed during the detection evaluation"
        raise RuntimeError(message)
    results = {
        "scope": lock["scope"],
        "source_groups": settings["test_source_groups"],
        "source_overlap_with_pretraining": 0,
        "test_windows": len(windows),
        "empty_test_windows": int((np.diff(windows.offsets) == 0).sum()),
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory": peaks,
        "device": str(scorer.device),
        "gpu_name": torch.cuda.get_device_name(scorer.device)
        if scorer.device.type == "cuda"
        else None,
        "ensemble_parameters": sum(
            sum(p.numel() for p in model.parameters()) for model in scorer.models
        ),
        "operating_points": summarize_detection(probabilities, truth, config),
        "uncertainty": (
            "95% source-group bootstrap conditional on fixed models/calibration; "
            "Wilson null intervals; no multiplicity correction"
        ),
        "self_splitting_objective_isolated": False,
        "real_subthreshold_events_validated": False,
        "physical_parameters_decoded": False,
    }
    write_json(output / "test_results.json", results)
    _write_table(output / "detection_results.csv", results["operating_points"])
    create_report(output, config, results, probabilities, truth, windows, scorer)
    LOGGER.info("Completed frozen detection feasibility study: %s", output / "index.html")
    return results


def _write_table(path, operating_points):
    """Expose every operating point and paired recovery/miss count as a plain table."""
    records = [
        {
            "target_false_positive_rate": alpha,
            "method": method,
            "test_false_positive_rate": entry["false_positive_rate"],
            **row,
        }
        for alpha, methods in operating_points.items()
        for method, entry in methods.items()
        for row in entry["efficiency"]
    ]
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(records)


def _figure(output, figure, name):
    for extension in ("png", "svg"):
        figure.savefig(output / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def create_report(output, config, results, probabilities, truth, windows, scorer):
    """Export all methods, false alarms and deterministic injection-GT examples."""
    levels = config["generator"]["signal_expected_photons"]
    alpha = config["evaluation"]["primary_false_positive_rate"]
    primary = results["operating_points"][str(alpha)]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for (name, label), color in zip(LABELS.items(), COLORS, strict=True):
        rows = primary[name]["efficiency"]
        curve = np.asarray([row["efficiency"] for row in rows])
        intervals = np.asarray([row["interval"] for row in rows])
        axes[0].plot(levels, curve, "o-", label=label, color=color, markersize=4)
        axes[0].fill_between(levels, intervals[:, 0], intervals[:, 1], alpha=0.1, color=color)
        if name != "self_split":
            delta = np.asarray([row["model_minus_control"] for row in rows])
            ci = np.asarray([row["difference_interval"] for row in rows])
            axes[1].plot(levels, delta, "o-", label=f"Model minus {label.lower()}", color=color)
            axes[1].fill_between(levels, ci[:, 0], ci[:, 1], alpha=0.1, color=color)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Expected injected signal photons / window")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Detection efficiency")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylabel("Paired efficiency difference")
    axes[1].axhline(0, color="black", linewidth=0.7)
    figure.suptitle(f"Synthetic injection/recovery: target false-positive rate {alpha:g} / window")
    _figure(output, figure, "detection_efficiency")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    names = list(LABELS)
    for offset, rate in enumerate(config["evaluation"]["false_positive_rates"]):
        entries = results["operating_points"][str(rate)]
        y = np.asarray([entries[name]["false_positive_rate"] for name in names])
        ci = np.asarray([entries[name]["false_positive_interval"] for name in names])
        axes[0].errorbar(
            np.arange(len(names)) + (offset - 0.5) * 0.14,
            y,
            yerr=[y - ci[:, 0], ci[:, 1] - y],
            fmt="o",
            label=f"Target {rate:g}",
        )
        axes[0].axhline(rate, linewidth=0.7, linestyle="--")
    axes[0].set_xticks(
        np.arange(len(names)), [LABELS[name] for name in names], rotation=28, ha="right"
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Independent test false positives / window (Wilson 95%)")
    axes[0].legend()
    grid = np.geomspace(0.0001, 0.1, 180)
    null = truth["expected_signal"] == 0
    signal = truth["expected_signal"] == config["evaluation"]["primary_expected_signal"]
    for (name, label), color in zip(LABELS.items(), COLORS, strict=True):
        values = probabilities[name]
        fpr = np.asarray([(values[null] <= level).mean() for level in grid])
        tpr = np.asarray([(values[signal] <= level).mean() for level in grid])
        axes[1].plot(fpr, tpr, label=label, color=color)
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.0001, 0.1)
    axes[1].set_xlabel("Measured test false-positive rate / window")
    strength = config["evaluation"]["primary_expected_signal"]
    axes[1].set_ylabel(f"Efficiency at expected signal {strength:g} photons")
    axes[1].legend(fontsize=8)
    figure.suptitle("False alarms and low-FPR ROC: toy background, no detector exposure claim")
    _figure(output, figure, "false_alarms_roc")
    _examples_figure(output, config, truth, windows, scorer)
    rows = []
    for name, label in LABELS.items():
        point = next(
            row
            for row in primary[name]["efficiency"]
            if row["expected_signal"] == config["evaluation"]["primary_expected_signal"]
        )
        entry = primary[name]
        ci = entry["false_positive_interval"]
        delta = point.get("model_minus_control")
        interval = point.get("difference_interval")
        difference = (
            "&mdash;" if delta is None else f"{delta:+.3f} [{interval[0]:+.3f}, {interval[1]:+.3f}]"
        )
        rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{entry['false_positive_rate']:.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}]</td><td>{point['efficiency']:.3f}</td>"
            f"<td>{difference}</td></tr>"
        )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Subthreshold detection feasibility</title><style>"
        "body{max-width:1200px;margin:35px auto;padding:20px;font:17px system-ui;"
        "line-height:1.5;color:#213345}img{max-width:100%}"
        "td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}"
        "table{border-collapse:collapse}a{color:#176b95}"
        ".note{background:#fff0d5;padding:16px}</style>"
        "<h1>Subthreshold detection: frozen-model feasibility</h1>"
        '<p class="note"><b>Exact Poisson toy only.</b> These results do not establish '
        "recovery of real neutrino events, direction/energy decoding, or a causal benefit "
        "of self-splitting. The previously completed predictive studies remain unchanged.</p>"
        f"<p>{len(windows):,} fresh test windows; "
        f"{config['generator']['test_background_windows']:,} background-only windows and "
        f"{results['source_groups']} unseen signal-source groups. "
        f"Background mean {config['generator']['background_expected_photons']:g} photon(s); "
        "full recorded photon budget for all methods. "
        "Models and calibration were locked before test generation.</p>"
        "<h2>Primary operating point</h2>"
        f"<p>Target false-positive rate {alpha:g} per window; "
        f"expected injected signal {strength:g} photons. "
        "Brackets are descriptive 95% intervals. Source intervals condition on the frozen "
        "calibration/models; actual test false-alarm rates vary.</p>"
        "<table><tr><th>Method</th><th>Test false-positive rate</th>"
        "<th>Detection efficiency</th><th>Model minus control</th></tr>"
        + "".join(rows)
        + "</table>"
        "<p>Count ties use independent randomized ranks, with the same randomizer for each "
        f"method. The template control searches a fixed {len(scorer.template_sources)}-source "
        "lattice and includes its search penalty through background calibration.</p>"
        '<h2>Efficiency and paired comparisons</h2><a href="detection_efficiency.svg">'
        '<img src="detection_efficiency.png" '
        'alt="Efficiency and paired differences by expected signal"></a>'
        '<h2>False alarms and ROC</h2><a href="false_alarms_roc.svg">'
        '<img src="false_alarms_roc.png" '
        'alt="Independent background false alarms and low false-positive ROC"></a>'
        "<h2>Injection truth and recorded observations</h2>"
        "<p>Cases are the first event IDs at fixed expected strengths, chosen without "
        "checking detection success. GT is a known expected arrival distribution, "
        "not a clean photon image. Both recorded halves contribute to detection; "
        'the displayed model PDF uses A only.</p><a href="injection_examples.svg">'
        '<img src="injection_examples.png" '
        'alt="Expected analytic distributions, recorded A/B counts and predictions"></a>'
        '<p><a href="test_results.json">All results, including misses and recoveries</a> &middot; '
        '<a href="detection_results.csv">Download the comparison table</a> &middot; '
        '<a href="protocol.txt">Phased protocol and real-data requirements</a> &middot; '
        '<a href="analysis_lock.json">Frozen calibration/model/code identity</a> &middot; '
        '<a href="../study/results_index.html">Existing results hub</a></p></html>'
    )
    (output / "index.html").write_text(document, encoding="utf-8")


def _examples_figure(output, config, truth, windows, scorer):
    settings = config["generator"]["signal_expected_photons"]
    levels = (settings[1], config["evaluation"]["primary_expected_signal"], settings[-2])
    figure, axes = plt.subplots(len(levels), 4, figsize=(14, 8.5), layout="constrained")
    norm = LogNorm(1e-7, 0.03)
    background = config["generator"]["background_expected_photons"]
    for row, level in enumerate(levels):
        index = int(np.flatnonzero(truth["expected_signal"] == level)[0])
        a, b = windows.dense_batch(index, index + 1, scorer.shape)
        predicted = scorer.predict(a)[0]
        source = truth["sources_m"][truth["source_group"][index]]
        pdf = (level * scorer.signal_pdf(source) + background / scorer.prior.size) / (
            level + background
        )
        for column, (title, values) in enumerate(
            (
                ("Expected arrival PDF (evaluation GT)", pdf),
                ("Recorded A counts", a[0]),
                ("Recorded B counts", b[0]),
                ("Frozen model q(A)", predicted),
            )
        ):
            kwargs = (
                {"norm": norm, "cmap": "magma"}
                if column in (0, 3)
                else {"vmin": 0, "cmap": "Blues"}
            )
            rendered = axes[row, column].imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=[0, 6.4, -0.5, scorer.shape[0] - 0.5],
                **kwargs,
            )
            if row == 0:
                axes[row, column].set_title(title, fontsize=9)
            axes[row, column].set_xlabel("Time (microseconds)")
            if column == 0:
                axes[row, column].set_ylabel(
                    f"Expected signal {level:g}; ID {index}\nA={a.sum()}, B={b.sum()}\nSensor"
                )
            if column in (0, 3):
                figure.colorbar(
                    rendered, ax=axes[row, column], shrink=0.7, label="Probability / bin"
                )
    figure.suptitle(
        "Deterministic examples: latent expected distributions and recorded sparse photons"
    )
    _figure(output, figure, "injection_examples")


def main():
    """Run once into a fresh directory; existing results are never overwritten."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/neutrino_detection.yml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = yaml.safe_load(Path(args.config).read_text())
    run_detection(config, args.device)


if __name__ == "__main__":
    main()
