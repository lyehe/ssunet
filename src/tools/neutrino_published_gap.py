"""Compare the released GAP U-Net with the frozen 4D photon predictor."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from . import neutrino_gap4d_retry as retry
from .neutrino_conv4d_retry import write_json

if __package__ == "src.tools":
    from ..ssunet.models.neutrino_published_gap import PublishedGAPReference
    from ..ssunet.neutrino_training import _claim_run
else:
    from ssunet.models.neutrino_published_gap import PublishedGAPReference
    from ssunet.neutrino_training import _claim_run

LOGGER = logging.getLogger(__name__)


def make_model(settings, positions):
    """Use the unchanged released U-Net at the declared paper dimensions."""
    return PublishedGAPReference(
        positions,
        depth=settings["depth"],
        start_filts=settings["start_filts"],
        levels=settings["levels"],
    )


def make_optimizer(model, settings):
    """Keep the paper optimizer and the matched-optimizer control explicit."""
    options = settings["optimizer"]
    choices = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    optimizer = choices[options["kind"]](
        model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"]
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=options["scheduler_patience"]
        )
        if options["scheduler_patience"] is not None
        else None
    )
    return optimizer, scheduler


def normalized_validation_loss(losses, view):
    """Recover GAP's mean-target loss from summed photon cross entropies."""
    locations = int(view.live.sum()) * view.inputs.shape[-1]
    totals = view.target_totals
    return float(np.mean((losses - totals * math.log(locations)) / (totals + 1e-8 * locations)))


def _evaluate_epoch(model, optimizer, scheduler, state, config, validation, diagnostic, device):
    """Step the paper scheduler on its validation loss once per complete data epoch."""
    step = state["step"]
    is_curve = step in config["evaluation"]["steps"] or step == config["training"]["steps"]
    is_epoch = step % state["updates_per_epoch"] == 0
    losses, row = None, None
    if is_curve:
        row, losses = retry.curve_point(model, validation, diagnostic, device, config, step)
    if is_epoch and scheduler is not None:
        if losses is None:
            losses = retry.reference.model_event_nll(
                model, validation, device, config["training"]["batch_size"]
            )
        metric = normalized_validation_loss(losses, validation)
        scheduler.step(metric)
        state["scheduler_state_dict"] = scheduler.state_dict()
        state["epoch_metrics"].append(
            {
                "step": step,
                "validation_loss": metric,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
    if row is not None:
        row["learning_rate"] = optimizer.param_groups[0]["lr"]
        state["curve"].append(row)
        LOGGER.info(
            "%s seed=%s step=%s low=%.4f event_low=%.4f lr=%.3g training_s=%.1f",
            state["model_name"],
            state["seed"],
            step,
            row["retained_1_to_10"],
            row["event_mean_low"],
            row["learning_rate"],
            state["training_seconds"],
        )
    return is_curve


def train_seed(config, settings, seed, training, validation, diagnostic, root, device):
    """Train exactly the fixed data budget, retaining full resumable scheduler state."""
    directory = Path(root) / settings["name"] / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    run_config = copy.deepcopy(config)
    run_config["training"].update(settings.get("training_overrides", {}))
    run_config["training"]["gradient_clip"] = settings["gradient_clip"]
    options = run_config["training"]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = make_model(settings, training.positions_m).to(device)
    optimizer, scheduler = make_optimizer(model, settings)
    state = retry._restore_or_start(
        directory, model, optimizer, run_config, settings, seed, device, (validation, diagnostic)
    )
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    updates_per_epoch = (len(training) + options["batch_size"] - 1) // options["batch_size"]
    state.update(
        {"updates_per_epoch": updates_per_epoch, "model_name": settings["name"], "seed": seed}
    )
    state.setdefault("epoch_metrics", [])
    sampler = retry.reference.EpochPhotonBatchSampler(
        len(training), options["batch_size"], seed=seed
    )
    sampler.epoch, sampler.start_batch = divmod(state["step"], updates_per_epoch)
    loader = DataLoader(
        training,
        batch_sampler=sampler,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    started = time.perf_counter()
    while state["step"] < options["steps"]:
        iterator = iter(loader)
        for _ in range(len(loader)):
            if state["training_seconds"] >= options["max_training_seconds_per_seed"]:
                message = "Published baseline reached its declared training-time ceiling"
                raise RuntimeError(message)
            retry.reference.synchronize(device)
            begin = time.perf_counter()
            retry.training_step(model, optimizer, next(iterator), device, run_config)
            retry.reference.synchronize(device)
            state["training_seconds"] += time.perf_counter() - begin
            state["step"] += 1
            is_curve = _evaluate_epoch(
                model, optimizer, scheduler, state, run_config, validation, diagnostic, device
            )
            if state["step"] % 250 == 0 or is_curve:
                retry.save_checkpoint(
                    directory, model, optimizer, state, run_config, settings, seed, device
                )
            if state["step"] == options["steps"]:
                break
        sampler.epoch += 1
        sampler.start_batch = 0
    if state["peak_gpu_allocated_gib"] > options["max_gpu_allocated_gib"]:
        message = "Published baseline exceeded its declared GPU allocation ceiling"
        raise RuntimeError(message)
    losses = retry.reference.model_event_nll(model, validation, device, options["batch_size"])
    np.savez_compressed(directory / "scores.npz", model=losses)
    result = {
        "model": settings["name"],
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": state["step"],
        "batch_size": options["batch_size"],
        "training_examples": state["step"] * options["batch_size"],
        "epochs": state["step"] / updates_per_epoch,
        "completed_fixed_schedule": state["step"] == options["steps"],
        "training_seconds": state["training_seconds"],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gib": state["peak_gpu_allocated_gib"],
        "peak_process_rss_gib": state["peak_process_rss_gib"],
        "curve": state["curve"],
        "epoch_metrics": state["epoch_metrics"],
        "final_learning_rate": optimizer.param_groups[0]["lr"],
    }
    write_json(directory / "result.json", result)
    return result, losses


def study_identity(config):
    """Bind published source, observations and the already completed 4D comparator."""
    result = retry.identity(config)
    source = Path(__file__).parents[1]
    if (
        retry.reference.shard_sha256(source / "ssunet/vendor/gap/GAP_UNET_ResBlock.py")
        != config["upstream_source_sha256"]
    ):
        message = "Released GAP network differs from its pinned upstream source"
        raise ValueError(message)
    paths = [Path(__file__)] + [
        source / name
        for name in (
            "ssunet/models/neutrino_published_gap.py",
            "ssunet/vendor/gap/GAP_UNET_ResBlock.py",
            "ssunet/vendor/gap/LICENSE",
            "ssunet/vendor/gap/provenance.json",
            "ssunet/vendor/gap/__init__.py",
            "ssunet/vendor/__init__.py",
        )
    ]
    result["published_source_sha256"] = {
        str(path): retry.reference.shard_sha256(path) for path in paths
    }
    previous = Path(config["reference_study"])
    result["reference_artifact_sha256"] = {
        name: retry.reference.shard_sha256(previous / name)
        for name in (
            "config.json",
            "identity.json",
            "results.json",
            "per_event_scores.npz",
            "evidence_audit.json",
        )
    }
    return result


def _validate_reference(config, validation):
    """Require exactly the same observation IDs and photon splits as the 4D comparison."""
    root = Path(config["reference_study"])
    previous = json.loads((root / "config.json").read_text())
    if retry.identity(previous) != json.loads((root / "identity.json").read_text()):
        message = "The 4D comparator no longer matches its frozen source/data identity"
        raise ValueError(message)
    if previous["training"]["seeds"] != config["training"]["seeds"]:
        message = "Published comparison changes the paired model seeds"
        raise ValueError(message)
    for key in ("dataset_sha256", "split_probability", "photon_split_seed", "observation_kind"):
        if previous[key] != config[key]:
            message = "Published comparison changes the reference observation contract"
            raise ValueError(message)
    examples = previous["training"]["steps"] * previous["training"]["batch_size"]
    for settings in config["models"]:
        options = {**config["training"], **settings.get("training_overrides", {})}
        if options["steps"] * options["batch_size"] != examples:
            message = "Published comparison changes the fixed training-example budget"
            raise ValueError(message)
    with np.load(root / "per_event_scores.npz") as saved:
        expected = {
            "event_ids": validation.event_ids,
            "configuration_ids": validation.configuration_ids,
            "retained_totals": validation.inputs.sum((1, 2)).numpy(),
            "target_totals": validation.target_totals,
        }
        if any(not np.array_equal(saved[key], value) for key, value in expected.items()):
            message = "Validation observations or A/B splits differ from the frozen 4D comparator"
            raise ValueError(message)


def summarize(config, validation, runs, published_scores):
    """Report every declared published arm with conditional, paired configuration intervals."""
    with np.load(Path(config["reference_study"]) / "per_event_scores.npz") as saved:
        scores = {
            name: saved[name].copy()
            for name in ("conv4d", "full_time_reference", "previously_tuned_mixture")
        }
    scores.update({name: np.asarray(value) for name, value in published_scores.items()})
    means = {name: value.mean(0) if value.ndim == 2 else value for name, value in scores.items()}
    report = {
        "scope": config["scope"],
        "test_or_truth_used": False,
        "primary": "fixed-final-step retained_1_to_10 photon NLL",
        "scores": {
            name: retry.reference.photon_score_summary(value, validation)
            for name, value in means.items()
        },
        "event_mean_nll": {
            name: {
                stratum: retry._event_mean(value, validation.target_totals, mask)
                for stratum, mask in validation.strata.items()
            }
            for name, value in means.items()
        },
        "conv4d_minus_published": {
            name: {
                stratum: retry.reference.paired_photon_interval(
                    means["conv4d"] - value,
                    validation,
                    mask,
                    seed=config["evaluation"]["bootstrap_seed"],
                    draws=config["evaluation"]["bootstrap_draws"],
                )
                for stratum, mask in validation.strata.items()
            }
            for name, value in means.items()
            if name in published_scores
        },
        "runs": runs,
        "limitations": config["adaptations_and_limits"],
        "published_upstream_commit": config["upstream_commit"],
        "seed_aggregation": "mean event NLL across seeds, not prediction ensembling",
    }
    return report, scores


def run_study(config, device_name="cuda:0", *, resume=False):
    """Train both predeclared optimizer arms while preserving previous experiments."""
    retry.validate_method(config)
    if config["method"] != "gap" or config["training"]["precision"] != "fp32":
        message = "The released GAP count baseline uses the declared FP32 photon-count protocol"
        raise ValueError(message)
    root = Path(config["output"])
    current = study_identity(config)
    if current["dataset_sha256"] != config["dataset_sha256"]:
        message = "Dataset hash differs from the declared observation source"
        raise ValueError(message)
    if root.exists():
        if not resume or json.loads((root / "identity.json").read_text()) != current:
            message = "Existing published comparison requires --resume and identical sources"
            raise ValueError(message)
    else:
        root.mkdir(parents=True)
        write_json(root / "identity.json", current)
        write_json(root / "config.json", config)
    _claim_run(root)
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    training, _, validation, diagnostic, _ = retry.development_views(config)
    _validate_reference(config, validation)
    scores, runs = {}, []
    for settings in config["models"]:
        scores[settings["name"]] = []
        for seed in config["training"]["seeds"]:
            directory = root / settings["name"] / f"seed_{seed}"
            if (directory / "result.json").exists():
                result = json.loads((directory / "result.json").read_text())
                with np.load(directory / "scores.npz") as saved:
                    losses = saved["model"].copy()
            else:
                result, losses = train_seed(
                    config, settings, seed, training, validation, diagnostic, root, device
                )
            runs.append(result)
            scores[settings["name"]].append(losses)
            write_json(root / "progress.json", {"completed_runs": runs})
    report, all_scores = summarize(config, validation, runs, scores)
    write_json(root / "results.json", report)
    np.savez_compressed(
        root / "per_event_scores.npz",
        event_ids=validation.event_ids,
        configuration_ids=validation.configuration_ids,
        retained_totals=validation.inputs.sum((1, 2)).numpy(),
        target_totals=validation.target_totals,
        **all_scores,
    )
    training.close()
    write_json(root / "process.json", {"status": "complete", "completed_runs": len(runs)})
    return report


def main():
    """Run the pinned published-baseline comparison or resume a stopped worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/neutrino_published_gap.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_study(yaml.safe_load(Path(args.config).read_text()), args.device, resume=args.resume)


if __name__ == "__main__":
    main()
