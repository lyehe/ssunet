"""Recover the frozen 4D retry after an interrupted worker, preserving finished runs."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from . import neutrino_conv4d_retry as original
from .neutrino_conv4d_visualize import render_results

LOGGER = logging.getLogger(__name__)


def verify_identity(root, config):
    """Reject changed observations or original model/training/native implementations."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if original.shard_sha256(config["dataset"]) != manifest["dataset_sha256"]:
        message = "Dataset changed since the original run"
        raise ValueError(message)
    for name in ("source_sha256", "torch_convnd_source_sha256"):
        for path, expected in manifest[name].items():
            if original.shard_sha256(path) != expected:
                message = f"Frozen source changed: {path}"
                raise ValueError(message)
    return manifest


def logged_training_seconds(log_path, model, seed, step):
    """Recover the rounded completed-checkpoint timing without inventing lost work."""
    raw = log_path.read_bytes()
    text = raw.decode("utf-16" if raw.startswith(b"\xff\xfe") else "utf-8")
    pattern = rf"{re.escape(model)} seed={seed} step={step}\b[^\n]*training_s=([0-9.]+)"
    matches = re.findall(pattern, text)
    if not matches:
        message = "No timing record matches the saved checkpoint"
        raise ValueError(message)
    return float(matches[-1])


def continue_arm(
    config,
    settings,
    seed,
    training,
    validation,
    diagnostic,
    donors,
    device,
    root,
    *,
    previous_seconds,
):
    """Restore optimizer/model and derive the exact next deterministic data batch."""
    directory = root / settings["name"] / f"seed_{seed}"
    checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    if (
        checkpoint["config"] != config
        or checkpoint["model"] != settings
        or checkpoint["seed"] != seed
    ):
        message = "Saved checkpoint does not match the fixed experiment"
        raise ValueError(message)
    options = config["training"]
    torch.manual_seed(seed)
    model = original.make_model(settings, training.positions_m).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"]
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"]
    resumed_from = step
    curve = json.loads((directory / "learning_curve.json").read_text())
    replay, losses = original.curve_point(
        model, validation, diagnostic, device, options["batch_size"], step
    )
    for name in ("all", "retained_1_to_10", "fixed_train"):
        if abs(replay[name] - curve[-1][name]) > 1e-5:
            message = "Checkpoint fails frozen-score replay before resuming"
            raise ValueError(message)
    sampler = original.EpochPhotonBatchSampler(len(training), options["batch_size"], seed=seed)
    epoch, batch_offset = divmod(
        step, (len(training) + options["batch_size"] - 1) // options["batch_size"]
    )
    sampler.start_batch = batch_offset
    loader = DataLoader(training, batch_sampler=sampler, num_workers=0)
    resumed_seconds = 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while step < options["steps"]:
        sampler.epoch = epoch
        iterator = iter(loader)
        for _ in range(len(loader)):
            if previous_seconds + resumed_seconds >= options["max_training_seconds_per_seed"]:
                message = "Recovered run exceeded its fixed training budget"
                raise RuntimeError(message)
            original.synchronize(device)
            begin = time.perf_counter()
            original.training_step(model, optimizer, next(iterator), device, options)
            original.synchronize(device)
            resumed_seconds += time.perf_counter() - begin
            step += 1
            if step in config["evaluation"]["steps"] or step == options["steps"]:
                row, losses = original.curve_point(
                    model, validation, diagnostic, device, options["batch_size"], step
                )
                curve.append(row)
                torch.save(
                    {
                        "model": settings,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "seed": seed,
                        "step": step,
                        "epoch": epoch,
                        "validation": row,
                        "recovery": {
                            "from_step": resumed_from,
                            "known_training_seconds": previous_seconds + resumed_seconds,
                        },
                    },
                    directory / "last.pt",
                )
                original.write_json(directory / "learning_curve.json", curve)
                LOGGER.info(
                    "%s seed=%s recovered step=%s all=%.4f low=%.4f training_s=%.1f",
                    settings["name"],
                    seed,
                    step,
                    row["all"],
                    row["retained_1_to_10"],
                    previous_seconds + resumed_seconds,
                )
            if step >= options["steps"]:
                break
        epoch += 1
        sampler.start_batch = 0
    choices, donor_view = donors
    donor_losses = np.mean(
        [
            original.model_event_nll(
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
        "training_seconds": previous_seconds + resumed_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "examples_per_training_second": step
        * options["batch_size"]
        / (previous_seconds + resumed_seconds),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else 0.0,
        "peak_process_rss_gib": getattr(
            psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss
        )
        / 2**30,
        "curve": curve,
        "recovery": {
            "from_step": resumed_from,
            "previous_timing_resolution_seconds": 0.1,
            "resource_peak_scope": "resumed segment",
            "discarded_uncheckpointed_work": "unknown; excluded from timing",
            "schedule": "original model/optimizer and epoch/batch cursor restored",
        },
    }
    original.write_json(directory / "result.json", result)
    return result, losses, donor_losses


def recover_retry(root, device_name="cuda:0"):
    """Skip completed arms, resume partial arms, and produce the original comparison."""
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    manifest = verify_identity(root, config)
    recovery = root / ("recovery_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    recovery.mkdir()
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    arguments = {
        "lazy": True,
        "seed": config["photon_split_seed"],
        "p": config["split_probability"],
    }
    training = original.NeutrinoPhotonDataset(config["dataset"], "train", **arguments)
    frozen_train = original.NeutrinoPhotonDataset(
        config["dataset"], "train", training=False, **arguments
    )
    frozen_validation = original.NeutrinoPhotonDataset(config["dataset"], "validation", **arguments)
    training_view = original.freeze_photon_view(frozen_train)
    validation = original.freeze_photon_view(frozen_validation)
    frozen_train.close()
    frozen_validation.close()
    selection = np.random.default_rng(config["evaluation"]["training_subset_seed"]).permutation(
        len(training)
    )[: config["evaluation"]["fixed_training_events"]]
    diagnostic = original.subset_view(training_view, selection)
    choices = original.matched_donor_indices(
        training_view.inputs.sum((1, 2)).numpy(),
        validation.inputs.sum((1, 2)).numpy(),
        seed=config["evaluation"]["donor_seed"],
        draws=config["evaluation"]["donor_draws"],
    )
    baselines = original.fit_pilot_baselines(training, validation, config["baselines"])
    preserved = {str(path): original.shard_sha256(path) for path in root.rglob("result.json")}
    original.write_json(
        recovery / "manifest.json",
        {
            "reason": "worker interrupted by daemon restart",
            "preserved_completed_results": preserved,
            "original_manifest_sha256": original.shard_sha256(root / "manifest.json"),
            "recovery_source_sha256": original.shard_sha256(__file__),
            "no_hyperparameter_or_selection_change": True,
        },
    )
    results, scores = [], {}
    for settings in config["models"]:
        name = settings["name"]
        scores[name], scores[name + "_donor"] = [], []
        for seed in config["training"]["seeds"]:
            directory = root / name / f"seed_{seed}"
            if (directory / "result.json").exists():
                result = json.loads((directory / "result.json").read_text())
                with np.load(directory / "scores.npz") as stored:
                    losses, donor_losses = stored["model"], stored["donor"]
            elif (directory / "last.pt").exists():
                checkpoint = torch.load(
                    directory / "last.pt", map_location="cpu", weights_only=True
                )
                (recovery / f"{name}_seed_{seed}_step_{checkpoint['step']}.pt").write_bytes(
                    (directory / "last.pt").read_bytes()
                )
                prior = checkpoint.get("recovery", {}).get("known_training_seconds")
                if prior is None:
                    prior = logged_training_seconds(
                        Path("experiments/neutrino_conv4d_retry_v1.log"),
                        name,
                        seed,
                        checkpoint["step"],
                    )
                result, losses, donor_losses = continue_arm(
                    config,
                    settings,
                    seed,
                    training,
                    validation,
                    diagnostic,
                    (choices, training_view),
                    device,
                    root,
                    previous_seconds=prior,
                )
            else:
                result, losses, donor_losses = original.train_arm(
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
            scores[name].append(losses)
            scores[name + "_donor"].append(donor_losses)
            original.write_json(root / "progress.json", {"completed_runs": results})
    report = original.summarize(
        config,
        validation,
        results,
        scores,
        baselines,
        manifest["donor_exact_match_fraction_by_stratum"],
    )
    report["recovery_note"] = (
        "An interrupted 4D seed resumed from its saved checkpoint after a daemon restart. "
        "Final model/optimizer update budgets are unchanged; discarded worker time is unmeasured."
    )
    original.write_json(root / "results.json", report)
    np.savez_compressed(
        root / "per_event_scores.npz",
        event_ids=validation.event_ids,
        configuration_ids=validation.configuration_ids,
        target_totals=validation.target_totals,
        retained_totals=validation.inputs.sum((1, 2)).numpy(),
        **{name: np.asarray(value) for name, value in {**baselines, **scores}.items()},
    )
    render_results(
        root, report, config, validation, training.positions_m, device, original.make_model
    )
    for path, expected in preserved.items():
        if original.shard_sha256(path) != expected:
            message = "A completed run changed during recovery"
            raise RuntimeError(message)
    training.close()
    original.write_json(recovery / "completed.json", {"completed_runs": len(results)})
    return report


def main():
    """Resume only an explicitly named, existing fixed retry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/neutrino/conv4d_retry_v1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    recover_retry(args.output, args.device)


if __name__ == "__main__":
    main()
