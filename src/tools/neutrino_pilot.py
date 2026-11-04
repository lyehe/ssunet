"""Run a fixed-budget, three-seed continuation of the dense photon pilot."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import platform
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

if __package__ == "src.tools":
    from ..ssunet.datasets.neutrino import NeutrinoPhotonDataset, shard_sha256
    from ..ssunet.models.neutrino_dense import DensePhotonReference
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
    from ssunet.datasets.neutrino import NeutrinoPhotonDataset, shard_sha256
    from ssunet.models.neutrino_dense import DensePhotonReference
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


def _write_json(path: Path, content) -> None:
    """Persist finite machine-readable measurements after each evaluation."""
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _training_step(model, optimizer, batch, device, settings) -> tuple[float, int, int]:
    """Optimize raw held-out count NLL with the original pilot's update rule."""
    model.train()
    counts, target = batch["input"].to(device), batch["target"].to(device)
    live = batch["live"].to(device)
    optimizer.zero_grad(set_to_none=True)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        logits = model(counts, batch["positions_m"].to(device), live, batch["p"].to(device))
        loss = photon_count_nll(logits, target, live)
    if not bool(torch.isfinite(loss)):
        message = "Non-finite photon training likelihood"
        raise RuntimeError(message)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return float(loss.detach()) * len(counts), int(target.sum()), len(counts)


def _curve_point(model, train_view, validation, device, batch_size, step, interval) -> dict:
    """Score the same frozen observations at every scheduled checkpoint."""
    scores = photon_score_summary(
        model_event_nll(model, validation, device, batch_size), validation
    )
    fixed_training = model_event_nll(model, train_view, device, batch_size)
    row = {
        "optimizer_step": step,
        "training_interval_nats_per_photon": interval[0] / interval[1] if interval[1] else None,
        "training_interval_target_photons": interval[1],
        "training_interval_events": interval[2],
        "fixed_training_nats_per_photon": float(
            fixed_training.sum() / train_view.target_totals.sum()
        ),
        "validation_nats_per_photon": scores["all"]["nats_per_photon"],
        "validation_low_count_nats_per_photon": scores["retained_1_to_10"]["nats_per_photon"],
        "validation_zero_input_nats_per_photon": scores["retained_0"]["nats_per_photon"],
    }
    return row


def _save_checkpoint(path, model, optimizer, config, seed, step, epoch, row) -> None:
    """Save inference weights and optimizer state with explicit data provenance."""
    torch.save(
        {
            "schema_version": 1,
            "model_class": "DensePhotonReference",
            "model_channels": config["model_channels"],
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "dataset_sha256": config["dataset_sha256"],
            "model_seed": seed,
            "optimizer_step": step,
            "training_epoch": epoch,
            "validation_scores": row,
            "scope": "inference_and_optimizer_audit; CLI_resume_not_implemented",
        },
        path,
    )


def _record_point(
    directory, model, optimizer, config, seed, step, epoch, views, device, interval, rows
):
    """Save each scheduled checkpoint and append its comparable learning scores."""
    row = _curve_point(model, *views, device, config["training"]["batch_size"], step, interval)
    previous = min((r["validation_low_count_nats_per_photon"] for r in rows), default=np.inf)
    rows.append(row)
    _save_checkpoint(
        directory / f"step_{step:06d}.pt", model, optimizer, config, seed, step, epoch, row
    )
    if row["validation_low_count_nats_per_photon"] < previous:
        _save_checkpoint(
            directory / "best_low_count.pt", model, optimizer, config, seed, step, epoch, row
        )
    _write_json(directory / "learning_curve.json", rows)
    with (directory / "learning_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info(
        "seed=%s step=%s train=%.4f validation=%.4f low_count=%.4f",
        seed,
        step,
        row["fixed_training_nats_per_photon"],
        row["validation_nats_per_photon"],
        row["validation_low_count_nats_per_photon"],
    )


def _evaluate_controls(model, view, donors, donor_choices, baselines, device, config) -> dict:
    """Evaluate fixed controls without fitting or changing validation photon draws."""
    batch_size = config["training"]["batch_size"]
    losses = dict(baselines)
    losses["model"] = model_event_nll(model, view, device, batch_size)
    losses["model_zero_input"] = model_event_nll(
        model, view, device, batch_size, torch.zeros_like(view.inputs)
    )
    losses["model_count_matched_donor"] = np.mean(
        [
            model_event_nll(model, view, device, batch_size, donors[choice])
            for choice in donor_choices
        ],
        axis=0,
    )
    donor_difference = (
        donors.sum(dim=(1, 2)).numpy()[donor_choices] - view.inputs.sum(dim=(1, 2)).numpy()
    )
    strata = {}
    comparisons = ("fixed_mixture", "previously_tuned_mixture", "model_count_matched_donor")
    for name, mask in view.strata.items():
        strata[name] = {
            "scores": {
                key: photon_score_summary(value, view)[name] for key, value in losses.items()
            },
            "exact_donor_count_match_fraction": float((donor_difference[:, mask] == 0).mean())
            if mask.any()
            else None,
            "model_minus_control": {
                key: paired_photon_interval(
                    losses["model"] - losses[key],
                    view,
                    mask,
                    seed=config["evaluation"]["bootstrap_seed"],
                    draws=config["evaluation"]["bootstrap_draws"],
                )
                for key in comparisons
            },
        }
    return {
        "strata": strata,
        "per_event": [
            {
                "event_id": int(event_id),
                "configuration_id": int(group),
                "retained_photons": int(number),
                "held_out_photons": int(target),
                "nll_by_method": {key: float(value[index]) for key, value in losses.items()},
            }
            for index, (event_id, group, number, target) in enumerate(
                zip(
                    view.event_ids,
                    view.configuration_ids,
                    view.inputs.sum(dim=(1, 2)).numpy(),
                    view.target_totals,
                    strict=True,
                )
            )
        ],
    }


def _train_seed(seed, training, views, donors, donor_choices, baselines, config, directory, device):
    """Train one unchanged backbone for the predeclared schedule and wall limit."""
    settings = config["training"]
    directory.mkdir()
    torch.manual_seed(seed)
    training.set_epoch(0)
    model = DensePhotonReference(config["model_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    loader = DataLoader(
        training,
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    iterator, epoch = iter(loader), 0
    rows, interval = [], [0.0, 0, 0]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    _record_point(
        directory, model, optimizer, config, seed, 0, epoch, views, device, interval, rows
    )
    training_seconds, last_step = 0.0, 0
    evaluation_steps = set(config["evaluation"]["steps"])
    for step in range(1, settings["steps"] + 1):
        if time.perf_counter() - start > settings["max_seconds_per_seed"]:
            break
        begin = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            training.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        values = _training_step(model, optimizer, batch, device, settings)
        training_seconds += time.perf_counter() - begin
        interval = [left + right for left, right in zip(interval, values, strict=True)]
        last_step = step
        if step in evaluation_steps:
            _record_point(
                directory,
                model,
                optimizer,
                config,
                seed,
                step,
                epoch,
                views,
                device,
                interval,
                rows,
            )
            interval = [0.0, 0, 0]
    if rows[-1]["optimizer_step"] != last_step:
        _record_point(
            directory,
            model,
            optimizer,
            config,
            seed,
            last_step,
            epoch,
            views,
            device,
            interval,
            rows,
        )
    final = _evaluate_controls(model, views[1], donors, donor_choices, baselines, device, config)
    best_checkpoint = torch.load(
        directory / "best_low_count.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(best_checkpoint["state_dict"])
    selected = _evaluate_controls(model, views[1], donors, donor_choices, baselines, device, config)
    reproduced = selected["strata"]["retained_1_to_10"]["scores"]["model"]["nats_per_photon"]
    if (
        abs(
            reproduced
            - best_checkpoint["validation_scores"]["validation_low_count_nats_per_photon"]
        )
        > 1e-5
    ):
        message = "Saved checkpoint does not reproduce its frozen validation score"
        raise RuntimeError(message)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    result = {
        "seed": seed,
        "optimizer_steps": last_step,
        "completed_fixed_schedule": last_step == settings["steps"],
        "training_seconds_excluding_evaluation": training_seconds,
        "total_seed_elapsed_seconds": time.perf_counter() - start,
        "training_events_per_second": sum(row["training_interval_events"] for row in rows)
        / training_seconds,
        "peak_allocated_gpu_gib_including_evaluation": peak,
        "memory_target_met": peak < settings["memory_target_gib"] if peak is not None else None,
        "best_low_count_optimizer_step": best_checkpoint["optimizer_step"],
        "learning_curve": rows,
        "final": final,
        "validation_selected": selected,
    }
    _write_json(directory / "results.json", result)
    return result


def _aggregate(results: list, validation, config) -> dict:
    """Separate variation between seeds from descriptive event-bootstrap intervals."""
    report = {}
    for selection in ("final", "validation_selected"):
        report[selection] = {}
        for stratum in ("all", "retained_0", "retained_1_to_10"):
            values = [
                row[selection]["strata"][stratum]["scores"]["model"]["nats_per_photon"]
                for row in results
            ]
            report[selection][stratum] = {
                "model_nats_per_photon_by_seed": values,
                "mean_seed_nats_per_photon": float(np.mean(values))
                if values[0] is not None
                else None,
                "sample_std_across_seeds": float(np.std(values, ddof=1))
                if values[0] is not None and len(values) > 1
                else None,
                "mean_seed_model_minus_control": {},
            }
            for control in (
                "fixed_mixture",
                "previously_tuned_mixture",
                "model_count_matched_donor",
            ):
                deltas = np.asarray(
                    [
                        [
                            event["nll_by_method"]["model"] - event["nll_by_method"][control]
                            for event in row[selection]["per_event"]
                        ]
                        for row in results
                    ]
                )
                report[selection][stratum]["mean_seed_model_minus_control"][control] = (
                    paired_photon_interval(
                        deltas.mean(axis=0),
                        validation,
                        validation.strata[stratum],
                        seed=config["evaluation"]["bootstrap_seed"],
                        draws=config["evaluation"]["bootstrap_draws"],
                    )
                )
    return report


def _plot_curves(output, results, baselines, validation) -> None:
    """Export learning curves using ordinary plotting tools."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    columns = (
        "fixed_training_nats_per_photon",
        "validation_nats_per_photon",
        "validation_low_count_nats_per_photon",
    )
    for axis, column, title in zip(
        axes,
        columns,
        (
            "Frozen training diagnostic",
            "Validation: all photons",
            "Validation: 1 to 10 retained photons",
        ),
        strict=True,
    ):
        for result in results:
            rows = result["learning_curve"]
            axis.plot(
                [row["optimizer_step"] for row in rows],
                [row[column] for row in rows],
                marker=".",
                label=f"Seed {result['seed']}",
            )
        axis.set_title(title)
        axis.set_xlabel("Optimizer steps")
        axis.set_xscale("symlog", linthresh=110)
        axis.set_ylabel("NLL (nats per target photon)")
        axis.grid(alpha=0.2)
    for axis, stratum in zip(axes[1:], ("all", "retained_1_to_10"), strict=True):
        for name, style in (("fixed_mixture", "--"), ("previously_tuned_mixture", ":")):
            value = photon_score_summary(baselines[name], validation)[stratum]["nats_per_photon"]
            axis.axhline(
                value, linestyle=style, color="black", alpha=0.65, label=name.replace("_", " ")
            )
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=7)
    figure.suptitle("Longer photon pilot: unchanged architecture and objective; validation only")
    for extension in ("png", "svg"):
        figure.savefig(output / f"learning_curves.{extension}", dpi=180)
    plt.close(figure)


def run_pilot(config: dict, output: Path, device_name: str = "cuda:0") -> dict:
    """Run the fixed pilot, refusing to overwrite earlier experiment artifacts."""
    output = Path(output)
    if output.exists():
        message = f"Refusing to overwrite an experiment directory: {output}"
        raise FileExistsError(message)
    settings = config["training"]
    if (
        settings["steps"] < 1
        or not settings["seeds"]
        or len(set(settings["seeds"])) != len(settings["seeds"])
    ):
        message = "Training requires positive steps and distinct nonempty model seeds"
        raise ValueError(message)
    path = Path(config["dataset"])
    if shard_sha256(path) != config["dataset_sha256"]:
        message = "Observation shard hash differs from the fixed experiment configuration"
        raise ValueError(message)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(settings["cpu_threads"])
    kwargs = {"p": config["split_probability"], "seed": config["photon_split_seed"]}
    training = NeutrinoPhotonDataset(path, "train", **kwargs)
    frozen_training = NeutrinoPhotonDataset(path, "train", training=False, **kwargs)
    validation = freeze_photon_view(NeutrinoPhotonDataset(path, "validation", **kwargs))
    if not validation.target_totals[validation.strata["retained_1_to_10"]].sum():
        message = "The primary low-count validation stratum contains no target photons"
        raise ValueError(message)
    if set(training.configuration_ids[training.indices]) & set(validation.configuration_ids):
        message = "A latent configuration crosses training and validation partitions"
        raise ValueError(message)
    evaluation = config["evaluation"]
    selected = np.random.default_rng(evaluation["training_subset_seed"]).permutation(
        len(frozen_training)
    )[: evaluation["fixed_training_events"]]
    train_view = freeze_photon_view(frozen_training, selected)
    donors = torch.stack([frozen_training[index]["input"] for index in range(len(frozen_training))])
    choices = matched_donor_indices(
        donors.sum(dim=(1, 2)).numpy(),
        validation.inputs.sum(dim=(1, 2)).numpy(),
        seed=evaluation["donor_seed"],
        draws=evaluation["donor_draws"],
    )
    baselines = fit_pilot_baselines(frozen_training, validation, config["baselines"])
    output.mkdir(parents=True)
    manifest = {
        "scope": "fixed_longer_pilot_same_architecture; exploratory_validation_only",
        "config": config,
        "test_partition_scored": False,
        "truth_sidecars_used": False,
        "checkpoint_selection": "minimum_low_count_validation_NLL_at_scheduled_checkpoints",
        "baseline_selection": "frozen_from_previous_pilot; no_new_hyperparameter_search",
        "training_diagnostic_event_ids": train_view.event_ids.tolist(),
        "validation_event_ids": validation.event_ids.tolist(),
        "validation_configuration_ids": validation.configuration_ids.tolist(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "BF16_training_FP32_normalization_and_evaluation"
        if device.type == "cuda"
        else "FP32",
        "code_sha256": {
            str(Path(inspect.getfile(obj))): shard_sha256(inspect.getfile(obj))
            for obj in (
                run_pilot,
                DensePhotonReference,
                NeutrinoPhotonDataset,
                photon_count_nll,
                model_event_nll,
            )
        },
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(
        output / "baseline_scores.json",
        {name: photon_score_summary(value, validation) for name, value in baselines.items()},
    )
    results = [
        _train_seed(
            seed,
            training,
            (train_view, validation),
            donors,
            choices,
            baselines,
            config,
            output / f"seed_{seed}",
            device,
        )
        for seed in settings["seeds"]
    ]
    summary = {**manifest, "across_seeds": _aggregate(results, validation, config), "runs": results}
    _write_json(output / "summary.json", summary)
    _plot_curves(output, results, baselines, validation)
    return summary


def main() -> None:
    """Run a fixed experiment from a versioned configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/neutrino_pilot.yml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_pilot(yaml.safe_load(args.config.read_text(encoding="utf-8")), args.output, args.device)


if __name__ == "__main__":
    main()
