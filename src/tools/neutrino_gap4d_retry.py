"""Run the detector-geometry comparison with data-selected GAP/Bit2Bit supervision."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch_convnd
import yaml
from torch.utils.data import DataLoader

from . import neutrino_conv4d_retry as reference
from .neutrino_conv4d_visualize import render_results

if __package__ == "src.tools":
    from ..ssunet.models.neutrino_gap import GAPGridPhotonReference4d, GAPTemporalPhotonReference
    from ..ssunet.neutrino_training import _claim_run
    from ..ssunet.photon_methods import method_for_observation, paper_photon_loss
else:
    from ssunet.models.neutrino_gap import GAPGridPhotonReference4d, GAPTemporalPhotonReference
    from ssunet.neutrino_training import _claim_run
    from ssunet.photon_methods import method_for_observation, paper_photon_loss

LOGGER = logging.getLogger(__name__)


def make_model(settings, positions):
    """Use identical ten-scale raw-count encoding in every architecture arm."""
    options = {"channels": settings["channels"], "encoding_levels": settings["encoding_levels"]}
    if settings["name"] == "conv4d":
        return GAPGridPhotonReference4d(
            positions, backend=settings.get("backend", "auto"), **options
        )
    if settings["name"] in {"dense_reference", "full_time_reference"}:
        return GAPTemporalPhotonReference(
            full_time=settings["name"] == "full_time_reference", **options
        )
    message = "Unknown declared architecture"
    raise ValueError(message)


def _event_mean(losses, totals, mask):
    """Score each nonempty observed target once, separately from photon weighting."""
    active = mask & (totals > 0)
    return float((losses[active] / totals[active]).mean()) if active.any() else None


def curve_point(model, validation, diagnostic, device, config, step):
    """Keep photon-weighted comparisons and add GAP-aligned per-event diagnostics."""
    row, losses = reference.curve_point(
        model, validation, diagnostic, device, config["training"]["batch_size"], step
    )
    row["event_mean_all"] = _event_mean(losses, validation.target_totals, validation.strata["all"])
    row["event_mean_low"] = _event_mean(
        losses, validation.target_totals, validation.strata["retained_1_to_10"]
    )
    return row, losses


def training_step(model, optimizer, batch, device, config):
    """Keep raw counts for the method-specific loss; normalize targets only there."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    counts, target = batch["input"].to(device), batch["target"].to(device)
    live = batch["live"].to(device)
    logits = model(counts, batch["positions_m"].to(device), live, batch["p"].to(device))
    loss = paper_photon_loss(
        logits, target, live, counts, observation_kind=config["observation_kind"]
    )
    if not torch.isfinite(loss):
        message = "Non-finite paper-aligned photon loss"
        raise RuntimeError(message)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()


def save_checkpoint(directory, model, optimizer, state, config, settings, seed, device):
    """Atomically retain the optimizer, deterministic batch cursor and RNG states."""
    state["peak_gpu_allocated_gib"] = max(
        state.get("peak_gpu_allocated_gib", 0.0),
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
    )
    state["peak_process_rss_gib"] = max(
        state.get("peak_process_rss_gib", 0.0),
        getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)
        / 2**30,
    )
    payload = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "state": state,
        "config": config,
        "model": settings,
        "seed": seed,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }
    temporary = directory / "last.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(directory / "last.pt")
    reference.write_json(directory / "learning_curve.json", state["curve"])


def _restore_or_start(directory, model, optimizer, config, settings, seed, device, views):
    """Resume complete saved state without redoing already optimized photon batches."""
    if (directory / "last.pt").exists():
        saved = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["model"] != settings or saved["seed"] != seed:
            message = "Saved run identity differs from this fixed experiment"
            raise ValueError(message)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
        return saved["state"]
    row, _ = curve_point(model, *views, device, config, 0)
    state = {"step": 0, "training_seconds": 0.0, "curve": [row]}
    save_checkpoint(directory, model, optimizer, state, config, settings, seed, device)
    return state


def train_seed(config, settings, seed, training, validation, diagnostic, donors, root, device):
    """Use the same fixed FP32 training budget with the selected published objective."""
    directory = root / settings["name"] / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = make_model(settings, training.positions_m).to(device)
    options = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"]
    )
    state = _restore_or_start(
        directory, model, optimizer, config, settings, seed, device, (validation, diagnostic)
    )
    sampler = reference.EpochPhotonBatchSampler(len(training), options["batch_size"], seed=seed)
    batches = (len(training) + options["batch_size"] - 1) // options["batch_size"]
    sampler.epoch, sampler.start_batch = divmod(state["step"], batches)
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
                message = "Fixed training time budget reached"
                raise RuntimeError(message)
            reference.synchronize(device)
            begin = time.perf_counter()
            training_step(model, optimizer, next(iterator), device, config)
            reference.synchronize(device)
            state["training_seconds"] += time.perf_counter() - begin
            state["step"] += 1
            if state["step"] in config["evaluation"]["steps"] or state["step"] == options["steps"]:
                row, _ = curve_point(model, validation, diagnostic, device, config, state["step"])
                state["curve"].append(row)
                LOGGER.info(
                    "%s %s seed=%s step=%s photon_low=%.4f event_low=%.4f training_s=%.1f",
                    config["method"],
                    settings["name"],
                    seed,
                    state["step"],
                    row["retained_1_to_10"],
                    row["event_mean_low"],
                    state["training_seconds"],
                )
            if (
                state["step"] % 250 == 0
                or state["step"] in config["evaluation"]["steps"]
                or state["step"] == options["steps"]
            ):
                save_checkpoint(directory, model, optimizer, state, config, settings, seed, device)
            if state["step"] == options["steps"]:
                break
        sampler.epoch += 1
        sampler.start_batch = 0
    losses = reference.model_event_nll(model, validation, device, options["batch_size"])
    choices, donor_view = donors
    donor_losses = np.mean(
        [
            reference.model_event_nll(
                model, validation, device, options["batch_size"], donor_view.inputs[pick]
            )
            for pick in choices
        ],
        axis=0,
    )
    np.savez_compressed(directory / "scores.npz", model=losses, donor=donor_losses)
    result = {
        "model": settings["name"],
        "method": config["method"],
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": state["step"],
        "completed_fixed_schedule": state["step"] == options["steps"],
        "training_seconds": state["training_seconds"],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gib": state["peak_gpu_allocated_gib"],
        "peak_process_rss_gib": state["peak_process_rss_gib"],
        "curve": state["curve"],
    }
    reference.write_json(directory / "result.json", result)
    return result, losses, donor_losses


def development_views(config):
    """Expose only train/validation counts; never open test observations or truth."""
    options = {"lazy": True, "seed": config["photon_split_seed"], "p": config["split_probability"]}
    training = reference.NeutrinoPhotonDataset(config["dataset"], "train", **options)
    frozen_training = reference.NeutrinoPhotonDataset(
        config["dataset"], "train", training=False, **options
    )
    frozen_validation = reference.NeutrinoPhotonDataset(config["dataset"], "validation", **options)
    train_view = reference.freeze_photon_view(frozen_training)
    validation = reference.freeze_photon_view(frozen_validation)
    frozen_training.close()
    frozen_validation.close()
    ids = np.random.default_rng(config["evaluation"]["training_subset_seed"]).permutation(
        len(training)
    )
    diagnostic = reference.subset_view(
        train_view, ids[: config["evaluation"]["fixed_training_events"]]
    )
    choices = reference.matched_donor_indices(
        train_view.inputs.sum((1, 2)).numpy(),
        validation.inputs.sum((1, 2)).numpy(),
        seed=config["evaluation"]["donor_seed"],
        draws=config["evaluation"]["donor_draws"],
    )
    return training, train_view, validation, diagnostic, choices


def identity(config):
    """Bind the normalization policy, implementations and observations before fitting."""
    source = Path(__file__).parents[1]
    paths = [Path(__file__)] + [
        source / path
        for path in (
            "ssunet/photon_methods.py",
            "ssunet/models/neutrino_gap.py",
            "ssunet/models/neutrino_conv4d.py",
            "ssunet/models/neutrino_dense.py",
            "ssunet/photon_reference.py",
            "ssunet/neutrino_evaluation.py",
            "ssunet/distributions.py",
            "ssunet/datasets/neutrino.py",
            "tools/neutrino_conv4d_retry.py",
        )
    ]
    native = [
        path
        for path in Path(torch_convnd.__file__).parent.rglob("*")
        if path.suffix in {".py", ".cpp", ".cu", ".h", ".hpp", ".cuh"}
    ]
    return {
        "config": config,
        "dataset_sha256": reference.shard_sha256(config["dataset"]),
        "source_sha256": {str(path): reference.shard_sha256(path) for path in paths},
        "torch_convnd_source_sha256": {str(path): reference.shard_sha256(path) for path in native},
        "torch_version": str(torch.__version__),
    }


def validate_method(config):
    """Refuse a method override that contradicts the declared data format."""
    if method_for_observation(config["observation_kind"]) != config["method"]:
        message = "Method conflicts with the declared acquisition format"
        raise ValueError(message)


def run_study(config, device_name="cuda:0", *, resume=False, render=True):
    """Fit exactly the declared data-appropriate method and preserve the earlier study."""
    validate_method(config)
    root = Path(config["output"])
    current = identity(config)
    if current["dataset_sha256"] != config["dataset_sha256"]:
        message = "Observed data hash differs from the fixed source"
        raise ValueError(message)
    if root.exists():
        if not resume or json.loads((root / "identity.json").read_text()) != current:
            message = "Existing results require --resume and unchanged data/configuration/source"
            raise ValueError(message)
    else:
        root.mkdir(parents=True)
        reference.write_json(root / "identity.json", current)
        reference.write_json(root / "config.json", config)
    _claim_run(root)
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    training, train_view, validation, diagnostic, choices = development_views(config)
    maximum = max(int(training.counts(index)[0].max()) for index in range(len(training)))
    if config["observation_kind"] == "one_bit" and maximum > 1:
        message = "Declared one-bit observations contain multi-photon bins"
        raise ValueError(message)
    matched = (
        train_view.inputs.sum((1, 2)).numpy()[choices]
        == validation.inputs.sum((1, 2)).numpy()[None]
    )
    coverage = {
        name: float(matched[:, mask].mean()) if mask.any() else None
        for name, mask in validation.strata.items()
    }
    baselines = reference.fit_pilot_baselines(training, validation, config["baselines"])
    reference.write_json(
        root / "manifest.json",
        {
            "method": config["method"],
            "observation_kind": config["observation_kind"],
            "train_max_bin_count": maximum,
            "train_events": len(training),
            "validation_events": len(validation.inputs),
            "test_or_truth_used": False,
            "time_bin_width_ns": float(training.edges_ns[1] - training.edges_ns[0]),
            "time_window_ns": [float(training.edges_ns[0]), float(training.edges_ns[-1])],
            "torch_version": str(torch.__version__),
            "donor_coverage": coverage,
            "normalization_reference": "GAP 135b8c6 / Bit2Bit 26ccc43; gradient checks pass",
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
                with np.load(directory / "scores.npz") as saved:
                    losses, donor_losses = saved["model"], saved["donor"]
            else:
                result, losses, donor_losses = train_seed(
                    config,
                    settings,
                    seed,
                    training,
                    validation,
                    diagnostic,
                    (choices, train_view),
                    root,
                    device,
                )
            results.append(result)
            scores[name].append(losses)
            scores[name + "_donor"].append(donor_losses)
            reference.write_json(root / "progress.json", {"completed_runs": results})
    report = reference.summarize(config, validation, results, scores, baselines, coverage)
    report["method"] = config["method"]
    report["normalization"] = {
        "input": config["input_encoding"],
        "target": config["target_normalization"],
        "output": "joint unit-sum live sensor/time PMF",
    }
    report["event_mean_nll"] = {
        name: {
            stratum: _event_mean(np.mean(values, axis=0), validation.target_totals, mask)
            for stratum, mask in validation.strata.items()
        }
        for name, values in {**scores, **{key: [value] for key, value in baselines.items()}}.items()
    }
    reference.write_json(root / "results.json", report)
    np.savez_compressed(
        root / "per_event_scores.npz",
        event_ids=validation.event_ids,
        configuration_ids=validation.configuration_ids,
        retained_totals=validation.inputs.sum((1, 2)).numpy(),
        target_totals=validation.target_totals,
        **{name: np.asarray(values) for name, values in {**baselines, **scores}.items()},
    )
    if render:
        render_results(root, report, config, validation, training.positions_m, device, make_model)
    training.close()
    reference.write_json(
        root / "process.json", {"status": "complete", "completed_runs": len(results)}
    )
    return report


def main():
    """Run or resume the explicitly selected paper-aligned photon study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/neutrino_gap4d_retry.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_study(yaml.safe_load(Path(args.config).read_text()), args.device, resume=args.resume)


if __name__ == "__main__":
    main()
