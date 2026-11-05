"""Retry the original observed-photon pilot with a full x/y/z/time CNN."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import psutil
import torch
import torch_convnd
import yaml
from torch.utils.data import DataLoader

if __package__ == "src.tools":
    from ..ssunet.datasets.neutrino import (
        EpochPhotonBatchSampler,
        NeutrinoPhotonDataset,
        shard_sha256,
    )
    from ..ssunet.models.neutrino_conv4d import GridPhotonReference4d
    from ..ssunet.models.neutrino_dense import DensePhotonReference
    from ..ssunet.models.neutrino_variants import FinePhotonReference
    from ..ssunet.neutrino_evaluation import (
        fit_pilot_baselines,
        freeze_photon_view,
        matched_donor_indices,
        model_event_nll,
        paired_photon_interval,
        photon_score_summary,
    )
    from ..ssunet.photon_reference import photon_count_nll
else:
    from ssunet.datasets.neutrino import (
        EpochPhotonBatchSampler,
        NeutrinoPhotonDataset,
        shard_sha256,
    )
    from ssunet.models.neutrino_conv4d import GridPhotonReference4d
    from ssunet.models.neutrino_dense import DensePhotonReference
    from ssunet.models.neutrino_variants import FinePhotonReference
    from ssunet.neutrino_evaluation import (
        fit_pilot_baselines,
        freeze_photon_view,
        matched_donor_indices,
        model_event_nll,
        paired_photon_interval,
        photon_score_summary,
    )
    from ssunet.photon_reference import photon_count_nll

LOGGER = logging.getLogger(__name__)


def write_json(path, value):
    """Replace only artifacts owned by this new experiment."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_model(settings, positions):
    """Select the three fixed architecture arms before observing their scores."""
    if settings["name"] == "dense_reference":
        return DensePhotonReference(settings["channels"])
    if settings["name"] == "full_time_reference":
        return FinePhotonReference(settings["channels"])
    if settings["name"] == "conv4d":
        return GridPhotonReference4d(
            positions, settings["channels"], backend=settings.get("backend", "auto")
        )
    message = "Unknown fixed architecture arm"
    raise ValueError(message)


def subset_view(view, indices):
    """Slice complete frozen records without resplitting any observed photons."""
    return replace(
        view,
        inputs=view.inputs[indices],
        targets=view.targets[indices],
        event_ids=view.event_ids[indices],
        configuration_ids=view.configuration_ids[indices],
    )


def synchronize(device):
    """Include completed GPU work in wall-clock measurements."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def training_step(model, optimizer, batch, device, settings):
    """Update FP32 logits using raw held-out counts, without clean rate targets."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    counts, target = batch["input"].to(device), batch["target"].to(device)
    live = batch["live"].to(device)
    logits = model(counts, batch["positions_m"].to(device), live, batch["p"].to(device))
    loss = photon_count_nll(logits, target, live)
    if not bool(torch.isfinite(loss)):
        message = "Non-finite FP32 training loss"
        raise RuntimeError(message)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()


def curve_point(model, validation, diagnostic, device, batch_size, step):
    """Score the same frozen train/validation splits at every checkpoint."""
    losses = model_event_nll(model, validation, device, batch_size)
    summary = photon_score_summary(losses, validation)
    train_losses = model_event_nll(model, diagnostic, device, batch_size)
    return {
        "step": step,
        "all": summary["all"]["nats_per_photon"],
        "retained_1_to_10": summary["retained_1_to_10"]["nats_per_photon"],
        "retained_0": summary["retained_0"]["nats_per_photon"],
        "fixed_train": float(train_losses.sum() / diagnostic.target_totals.sum()),
    }, losses


def train_arm(config, settings, seed, training, validation, diagnostic, donors, device, root):
    """Fit one seed with matched order, splits, FP32, optimizer and update budget."""
    directory = root / settings["name"] / f"seed_{seed}"
    directory.mkdir(parents=True)
    torch.manual_seed(seed)
    model = make_model(settings, training.positions_m).to(device)
    options = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"]
    )
    sampler = EpochPhotonBatchSampler(len(training), options["batch_size"], seed=seed)
    loader = DataLoader(training, batch_sampler=sampler, num_workers=0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    row, losses = curve_point(model, validation, diagnostic, device, options["batch_size"], 0)
    curve = [row]
    training_seconds, step, epoch = 0.0, 0, 0
    schedule = set(config["evaluation"]["steps"])
    started = time.perf_counter()
    while step < options["steps"]:
        sampler.epoch = epoch
        iterator = iter(loader)
        for _ in range(len(loader)):
            if training_seconds >= options["max_training_seconds_per_seed"]:
                break
            synchronize(device)
            begin = time.perf_counter()
            training_step(model, optimizer, next(iterator), device, options)
            synchronize(device)
            training_seconds += time.perf_counter() - begin
            step += 1
            if step in schedule or step == options["steps"]:
                row, losses = curve_point(
                    model, validation, diagnostic, device, options["batch_size"], step
                )
                curve.append(row)
                checkpoint = {
                    "model": settings,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "seed": seed,
                    "step": step,
                    "epoch": epoch,
                    "validation": row,
                }
                torch.save(checkpoint, directory / "last.pt")
                write_json(directory / "learning_curve.json", curve)
                LOGGER.info(
                    "%s seed=%s step=%s all=%.4f low=%.4f training_s=%.1f",
                    settings["name"],
                    seed,
                    step,
                    row["all"],
                    row["retained_1_to_10"],
                    training_seconds,
                )
            if step >= options["steps"]:
                break
        if training_seconds >= options["max_training_seconds_per_seed"]:
            break
        epoch += 1
    if curve[-1]["step"] != step:
        row, losses = curve_point(
            model, validation, diagnostic, device, options["batch_size"], step
        )
        curve.append(row)
        torch.save(
            {
                "model": settings,
                "state_dict": model.state_dict(),
                "config": config,
                "seed": seed,
                "step": step,
                "epoch": epoch,
            },
            directory / "last.pt",
        )
        write_json(directory / "learning_curve.json", curve)
    choices, donor_view = donors
    donor_losses = np.mean(
        [
            model_event_nll(
                model, validation, device, options["batch_size"], donor_view.inputs[pick]
            )
            for pick in choices
        ],
        axis=0,
    )
    np.savez_compressed(directory / "scores.npz", model=losses, donor=donor_losses)
    result = {
        "model": settings["name"],
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": step,
        "completed_fixed_schedule": step == options["steps"],
        "training_seconds": training_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "examples_per_training_second": step * options["batch_size"] / training_seconds,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else 0.0,
        "peak_process_rss_gib": getattr(
            psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss
        )
        / 2**30,
        "curve": curve,
    }
    if result["peak_gpu_allocated_gib"] > options["max_gpu_allocated_gib"]:
        message = "The run exceeded the predeclared GPU allocation budget"
        raise RuntimeError(message)
    write_json(directory / "result.json", result)
    return result, losses, donor_losses


def summarize(config, view, results, scores, baselines, donor_coverage):
    """Report seed means and paired configuration intervals without multiplying events."""
    means = {name: np.mean(values, axis=0) for name, values in scores.items()}
    controls = {**baselines, **means}
    summaries = {name: photon_score_summary(value, view) for name, value in controls.items()}
    candidate = means["conv4d"]
    comparisons = {
        "dense_reference",
        "full_time_reference",
        "fixed_mixture",
        "previously_tuned_mixture",
        "conv4d_donor",
    }
    contrasts = {
        name: {
            stratum: paired_photon_interval(
                candidate - value,
                view,
                mask,
                seed=config["evaluation"]["bootstrap_seed"],
                draws=config["evaluation"]["bootstrap_draws"],
            )
            if name != "conv4d_donor" or donor_coverage[stratum] == 1.0
            else None
            for stratum, mask in view.strata.items()
        }
        for name, value in controls.items()
        if name in comparisons
    }
    return {
        "scope": config["scope"],
        "dataset": config["dataset"],
        "test_or_truth_used": False,
        "primary": "fixed-final-step retained_1_to_10 photon NLL",
        "scores": summaries,
        "donor_exact_match_fraction_by_stratum": donor_coverage,
        "donor_interpretation": "Contrast intervals only for completely exact count-matched strata",
        "conv4d_minus_control": contrasts,
        "runs": results,
        "seed_aggregation": "mean event NLL, not probability ensembling or extra events",
        "limitations": [
            "Exploratory reuse of the original 100-event validation pilot; no fresh test claim.",
            "Curated P-ONE simulation; real untriggered background and Poisson physics unverified.",
            "Intervals condition on models, budget and earlier validation-selected baselines.",
            "Parameter counts and FLOPs differ; parameter counts are recorded per run.",
            "Empty inputs predict priors; empty targets contribute zero location loss.",
        ],
    }


def run_retry(config, device_name="cuda:0", *, render=True):
    """Create a separate, immutable run using only training/validation photon records."""
    root = Path(config["output"])
    if root.exists():
        message = "Retry output already exists; choose a new output directory"
        raise FileExistsError(message)
    if config["training"]["precision"] != "fp32":
        message = "This fixed retry implements matched FP32 training only"
        raise ValueError(message)
    if shard_sha256(config["dataset"]) != config["dataset_sha256"]:
        message = "Observed dataset hash differs from the frozen retry configuration"
        raise ValueError(message)
    device = torch.device(device_name)
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
    training = NeutrinoPhotonDataset(
        config["dataset"],
        "train",
        lazy=True,
        seed=config["photon_split_seed"],
        p=config["split_probability"],
    )
    training_view = freeze_photon_view(
        NeutrinoPhotonDataset(
            config["dataset"],
            "train",
            lazy=True,
            training=False,
            seed=config["photon_split_seed"],
            p=config["split_probability"],
        )
    )
    validation = freeze_photon_view(
        NeutrinoPhotonDataset(
            config["dataset"],
            "validation",
            lazy=True,
            seed=config["photon_split_seed"],
            p=config["split_probability"],
        )
    )
    selection = np.random.default_rng(config["evaluation"]["training_subset_seed"]).permutation(
        len(training)
    )[: config["evaluation"]["fixed_training_events"]]
    diagnostic = subset_view(training_view, selection)
    choices = matched_donor_indices(
        training_view.inputs.sum((1, 2)).numpy(),
        validation.inputs.sum((1, 2)).numpy(),
        seed=config["evaluation"]["donor_seed"],
        draws=config["evaluation"]["donor_draws"],
    )
    baselines = fit_pilot_baselines(training, validation, config["baselines"])
    donor_matched = (
        training_view.inputs.sum((1, 2)).numpy()[choices]
        == validation.inputs.sum((1, 2)).numpy()[None]
    )
    donor_coverage = {
        name: float(donor_matched[:, mask].mean()) if mask.any() else None
        for name, mask in validation.strata.items()
    }
    root.mkdir(parents=True)
    write_json(root / "config.json", config)
    source_root = Path(__file__).parents[1]
    paths = [Path(__file__)] + [
        source_root / path
        for path in (
            "ssunet/models/neutrino_conv4d.py",
            "ssunet/models/neutrino_dense.py",
            "ssunet/models/neutrino_variants.py",
            "ssunet/datasets/neutrino.py",
            "ssunet/neutrino_evaluation.py",
            "ssunet/photon_reference.py",
            "ssunet/distributions.py",
        )
    ]
    for path in paths:
        destination = root / "source_snapshot" / path.resolve().relative_to(source_root.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    native_sources = [
        path
        for path in Path(torch_convnd.__file__).parent.rglob("*")
        if path.suffix in {".py", ".cpp", ".cu", ".h", ".cuh", ".hpp"}
    ]
    write_json(
        root / "manifest.json",
        {
            "dataset_sha256": config["dataset_sha256"],
            "torch_version": str(torch.__version__),
            "torch_convnd_version": torch_convnd.__version__,
            "torch_convnd_source_sha256": {
                str(path): shard_sha256(path) for path in native_sources
            },
            "source_sha256": {str(path): shard_sha256(path) for path in paths},
            "train_events": len(training),
            "validation_events": len(validation.inputs),
            "time_bin_width_ns": float(training.edges_ns[1] - training.edges_ns[0]),
            "time_window_ns": [float(training.edges_ns[0]), float(training.edges_ns[-1])],
            "validation_event_ids": validation.event_ids.tolist(),
            "geometry_grid": list(GridPhotonReference4d(training.positions_m).grid_shape),
            "input_access": [
                "training observed photons",
                "validation observed photons",
                "fixed geometry",
            ],
            "test_or_truth_used": False,
            "donor_exact_match_fraction_by_stratum": donor_coverage,
        },
    )
    results, scores = [], {}
    for settings in config["models"]:
        scores[settings["name"]], scores[settings["name"] + "_donor"] = [], []
        for seed in config["training"]["seeds"]:
            result, losses, donor_losses = train_arm(
                config,
                settings,
                seed,
                training,
                validation,
                diagnostic,
                (choices, training_view),
                device,
                root,
            )
            results.append(result)
            scores[settings["name"]].append(losses)
            scores[settings["name"] + "_donor"].append(donor_losses)
            write_json(root / "progress.json", {"completed_runs": results})
            if not result["completed_fixed_schedule"]:
                write_json(root / "incomplete.json", result)
                message = "Resource budget prevented the matched fixed-step comparison"
                raise RuntimeError(message)
    report = summarize(config, validation, results, scores, baselines, donor_coverage)
    write_json(root / "results.json", report)
    np.savez_compressed(
        root / "per_event_scores.npz",
        event_ids=validation.event_ids,
        configuration_ids=validation.configuration_ids,
        target_totals=validation.target_totals,
        retained_totals=validation.inputs.sum((1, 2)).numpy(),
        **{name: np.asarray(value) for name, value in {**baselines, **scores}.items()},
    )
    if render:
        from .neutrino_conv4d_visualize import render_results

        render_results(root, report, config, validation, training.positions_m, device, make_model)
    training.close()
    return report


def main():
    """Run the fixed retry from a YAML configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/neutrino_conv4d_retry.yml")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_retry(yaml.safe_load(Path(args.config).read_text()), args.device)


if __name__ == "__main__":
    main()
