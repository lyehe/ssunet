"""Prepare and smoke-test a bounded, acquisition-held-out 3D/4D confocal GAP pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from filelock import FileLock

from ..ssunet.datasets.confocal import prepare_counts, sample_split, validate_patch_shape
from ..ssunet.models.confocal_gap import ConfocalGAPUNet, confocal_gap_loss
from ..ssunet.softmax_fix import fixed_confocal_gap_loss, fixed_log_softmax

LOGGER = logging.getLogger(__name__)


def load_config(path):
    """Reject incompatible observation modes or unsupported training settings early."""
    config = yaml.safe_load(Path(path).read_text())
    if config["observation_kind"] != "photon_counts" or config["method"] != "gap":
        message = "This confocal pilot is configured for GAP on true photon counts"
        raise ValueError(message)
    if config.get("logit_normalization", "legacy") not in {"legacy", "barron2026"}:
        message = "Unknown logit normalization"
        raise ValueError(message)
    train = config["training"]
    validate_patch_shape(train["patch_tzyx"])
    if (
        train["precision"] != "fp32"
        or train["optimizer"] != "Adam"
        or not 0 < config["split_probability"] < 1
        or any(train[name] < 1 for name in ("batch_size", "accumulation_steps", "steps"))
        or train["learning_rate"] <= 0
        or train["gradient_clip"] <= 0
        or config["evaluation"]["every_steps"] < 1
        or config["evaluation"]["patches"] < 1
    ):
        message = "Expected positive training/evaluation settings, FP32 and Adam"
        raise ValueError(message)
    sources = config["sources"]
    train_path = os.path.normcase(os.path.abspath(sources["train"]["path"]))
    validation_path = os.path.normcase(os.path.abspath(sources["validation"]["path"]))
    if train_path == validation_path:
        message = "Use separate source acquisitions for training and validation"
        raise ValueError(message)
    for settings in config["models"]:
        resolve_model_config(config, settings["name"])
    return config


def resolve_model_config(config, name):
    """Allow different physical batches while enforcing equal effective batches."""
    resolved = copy.deepcopy(config)
    options = resolved["training"]
    overrides = options.pop("model_batching", {}).get(name, {})
    if set(overrides) - {"batch_size", "accumulation_steps"}:
        message = "Model batching overrides may only change batch size and accumulation"
        raise ValueError(message)
    expected = options["batch_size"] * options["accumulation_steps"]
    options.update(overrides)
    if any(
        not isinstance(options[key], int) or isinstance(options[key], bool) or options[key] < 1
        for key in ("batch_size", "accumulation_steps")
    ):
        message = "Batch size and accumulation must be positive integers"
        raise ValueError(message)
    if options["batch_size"] * options["accumulation_steps"] != expected:
        message = "All models must use the same effective batch size"
        raise ValueError(message)
    return resolved


def _write_json(path, value):
    """Publish complete JSON documents through a same-directory atomic rename."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def prepare_data(config):
    """Keep network I/O out of the optimization loop and verify reusable caches."""
    counts, manifests = {}, {}
    for name in ("train", "validation"):
        destination = Path(config["cache"]) / f"{name}.npy"
        LOGGER.info("Preparing/verifying %s counts from %s", name, config["sources"][name]["path"])
        path, manifests[name] = prepare_counts(config["sources"][name], destination)
        counts[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        if any(
            want > have
            for want, have in zip(config["training"]["patch_tzyx"], counts[name].shape, strict=True)
        ):
            message = "Training patch exceeds a prepared acquisition crop"
            raise ValueError(message)
    return counts, manifests


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fixed_validation(counts, config, patches=None):
    """Freeze photon splits on held-out acquisition patches, independently of training."""
    return [
        sample_split(
            counts,
            config["training"]["patch_tzyx"],
            batch_size=1,
            p=config["split_probability"],
            seed=config["photon_split_seed"],
            index=index,
            stream=1,
        )
        for index in range(patches or config["evaluation"]["patches"])
    ]


@torch.inference_mode()
def evaluate(model, validation, device, *, normalization="legacy"):
    """Score arrival distributions using B photons that never enter model inputs."""
    model.eval()
    total_nll, photons, event_nll, mass_error = 0.0, 0, [], 0.0
    patch_nll, patch_photons = [], []
    for retained, target in validation:
        logits = model(retained.to(device))
        flat = logits.float().flatten(1)
        logq = fixed_log_softmax(flat) if normalization == "barron2026" else flat.log_softmax(-1)
        observed = target.to(device).flatten(1)
        nll = -(observed.double() * logq.double()).sum(-1)
        totals = observed.sum(-1)
        total_nll += float(nll.sum())
        photons += int(totals.sum())
        patch_nll.extend(nll.tolist())
        patch_photons.extend(totals.tolist())
        event_nll.extend((nll[totals > 0] / totals[totals > 0]).tolist())
        mass_error = max(mass_error, float((logq.exp().sum(-1) - 1).abs().max()))
    if not photons:
        message = "Validation has no held-out photons; choose a larger fixed crop"
        raise ValueError(message)
    return {
        "photon_nll": total_nll / photons,
        "patch_mean_nll": float(np.mean(event_nll)),
        "target_photons": photons,
        "uniform_photon_nll": math.log(math.prod(validation[0][0].shape[2:])),
        "max_probability_mass_error": mass_error,
        "per_patch_total_nll": patch_nll,
        "per_patch_target_photons": patch_photons,
    }


def training_batch(counts, config, step, microstep):
    """Address individual photon draws independently of physical batch grouping."""
    options = config["training"]
    batch_size = options["batch_size"]
    first_index = ((step - 1) * options["accumulation_steps"] + microstep) * batch_size
    pairs = [
        sample_split(
            counts,
            options["patch_tzyx"],
            batch_size=1,
            p=config["split_probability"],
            seed=config["photon_split_seed"],
            index=first_index + offset,
            stream=0,
        )
        for offset in range(batch_size)
    ]
    return tuple(torch.cat([pair[side] for pair in pairs], dim=0) for side in (0, 1))


def _optimizer_step(model, optimizer, counts, config, device, step):
    """Use fresh input-only splits and average the same patch loss in both modes."""
    options = config["training"]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for microstep in range(options["accumulation_steps"]):
        retained, target = training_batch(counts, config, step, microstep)
        retained, target = retained.to(device), target.to(device)
        objective = (
            fixed_confocal_gap_loss
            if config.get("logit_normalization") == "barron2026"
            else confocal_gap_loss
        )
        loss = objective(model(retained), target, retained)
        if not bool(torch.isfinite(loss)):
            message = "Nonfinite GAP training loss"
            raise RuntimeError(message)
        (loss / options["accumulation_steps"]).backward()
        loss_sum += float(loss.detach()) / options["accumulation_steps"]
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), options["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return loss_sum, float(norm)


def _source_hashes():
    """Record the implementation bytes used for this new, separate pilot."""
    root = Path(__file__).resolve().parents[2]
    files = [
        Path(__file__).resolve(),
        root / "src/ssunet/datasets/confocal.py",
        root / "src/ssunet/models/confocal_gap.py",
        root / "src/ssunet/photon_methods.py",
        root / "src/ssunet/distributions.py",
        root / "src/ssunet/photon_reference.py",
        root / "src/ssunet/softmax_fix.py",
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }


def _checkpoint(directory, model, optimizer, config, settings, state, device):
    temporary = directory / "last.pt.tmp"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "model": settings,
            "step": state["step"],
            "state": state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        },
        temporary,
    )
    temporary.replace(directory / "last.pt")


def _identity(config, settings, manifests, device, smoke):
    return {
        "config": config,
        "model": settings,
        "sources": manifests,
        "source_hashes": _source_hashes(),
        "torch_version": str(torch.__version__),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "scope": "capacity_smoke_test" if smoke else config["evaluation"]["scope"],
    }


def _restore_or_initialize(directory, model, optimizer, config, settings, validation, device):
    if (directory / "last.pt").exists():
        saved = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["model"] != settings or "state" not in saved:
            message = "Checkpoint configuration or format differs from this run"
            raise ValueError(message)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
        return saved["state"]
    row = evaluate(
        model, validation, device, normalization=config.get("logit_normalization", "legacy")
    )
    state = {
        "step": 0,
        "times": [],
        "losses": [],
        "norms": [],
        "curve": [{"step": 0, **row}],
        "peak_gpu_allocated_gib": 0.0,
        "peak_gpu_reserved_gib": 0.0,
    }
    _checkpoint(directory, model, optimizer, config, settings, state, device)
    _write_json(directory / "learning_curve.json", state["curve"])
    return state


def _update_memory(state, device):
    if device.type == "cuda":
        for name, getter in (
            ("peak_gpu_allocated_gib", torch.cuda.max_memory_allocated),
            ("peak_gpu_reserved_gib", torch.cuda.max_memory_reserved),
        ):
            state[name] = max(state[name], getter(device) / 2**30)


def _progress(directory, state, settings, steps, phase):
    _write_json(
        directory / "progress.json",
        {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "model": settings["name"],
            "phase": phase,
            "step": state["step"],
            "total_steps": steps,
            "training_seconds": sum(state["times"]),
            "last_loss": state["losses"][-1] if state["losses"] else None,
            "last_validation": state["curve"][-1],
            "peak_gpu_allocated_gib": state["peak_gpu_allocated_gib"],
            "peak_gpu_reserved_gib": state["peak_gpu_reserved_gib"],
        },
    )


def _train_schedule(
    directory, model, optimizer, counts, validation, config, settings, state, device, smoke
):
    options = config["training"]
    steps = 3 if smoke else options["steps"]
    source_hashes = _source_hashes()
    for step in range(state["step"] + 1, steps + 1):
        _synchronize(device)
        begin = time.perf_counter()
        loss, norm = _optimizer_step(model, optimizer, counts, config, device, step)
        _synchronize(device)
        state["times"].append(time.perf_counter() - begin)
        state["losses"].append(loss)
        state["norms"].append(norm)
        state["step"] = step
        _update_memory(state, device)
        if (
            state["peak_gpu_allocated_gib"] > options["max_gpu_allocated_gib"]
            or sum(state["times"]) > options["max_training_seconds"]
        ):
            message = "Configured GPU memory or training time bound exceeded"
            raise RuntimeError(message)
        if step == 1 or step % options.get("log_every_steps", 1) == 0 or step == steps:
            LOGGER.info(
                "%s step=%s/%s loss=%.6f seconds=%.3f peak_GiB=%.3f",
                settings["name"],
                step,
                steps,
                loss,
                state["times"][-1],
                state["peak_gpu_allocated_gib"],
            )
            _progress(directory, state, settings, steps, "training")
        evaluate_now = step == steps or (
            not smoke and step % config["evaluation"]["every_steps"] == 0
        )
        if evaluate_now:
            _progress(directory, state, settings, steps, "evaluating")
            scores = evaluate(
                model, validation, device, normalization=config.get("logit_normalization", "legacy")
            )
            state["curve"].append({"step": step, "training_seconds": sum(state["times"]), **scores})
            _write_json(directory / "learning_curve.json", state["curve"])
            LOGGER.info(
                "%s validation step=%s photon_NLL=%.6f",
                settings["name"],
                step,
                scores["photon_nll"],
            )
        if evaluate_now or step % options.get("checkpoint_every_steps", 25) == 0:
            if _source_hashes() != source_hashes:
                message = "Training source files changed during this run"
                raise RuntimeError(message)
            _checkpoint(directory, model, optimizer, config, settings, state, device)
    _update_memory(state, device)
    _progress(directory, state, settings, steps, "complete")


def _result(config, settings, model, state, smoke):
    options = config["training"]
    median = float(np.median(state["times"][1:] or state["times"]))
    effective_batch = options["batch_size"] * options["accumulation_steps"]
    return {
        "model": settings["name"],
        "smoke_test": smoke,
        "parameters": sum(p.numel() for p in model.parameters()),
        "channel_widths": list(model.channel_widths),
        "depth": len(model.channel_widths),
        "patch_tzyx": options["patch_tzyx"],
        "batch_size": options["batch_size"],
        "accumulation_steps": options["accumulation_steps"],
        "effective_batch_size": effective_batch,
        "logit_normalization": config.get("logit_normalization", "legacy"),
        "optimizer_steps": state["step"],
        "training_examples": state["step"] * effective_batch,
        "training_seconds": sum(state["times"]),
        "median_optimizer_step_seconds_excluding_first": median,
        "projected_500_update_training_seconds": median * 500,
        "peak_gpu_allocated_gib": state["peak_gpu_allocated_gib"],
        "peak_gpu_reserved_gib": state["peak_gpu_reserved_gib"],
        "training_losses": state["losses"],
        "gradient_norms_before_clip": state["norms"],
        "curve": state["curve"],
        "status": "passed",
        "limitation": "Three updates verify training execution and capacity only."
        if smoke
        else "Exploratory acquisition holdout; independent biological replicates are unverified.",
    }


def run_model(config, settings, counts, manifests, device, *, smoke, resume=False):
    """Run a bounded schedule with deterministic patch identities and resumable state."""
    config = resolve_model_config(config, settings["name"])
    suffix = f"smoke_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}" if smoke else "pilot"
    directory = Path(config["output"]) / suffix / settings["name"]
    directory.mkdir(parents=True, exist_ok=resume)
    identity = _identity(config, settings, manifests, device, smoke)
    identity_path = directory / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        message = "Saved source, configuration or data identity differs"
        raise ValueError(message)
    _write_json(identity_path, identity)
    if resume and (directory / "result.json").exists():
        return json.loads((directory / "result.json").read_text())
    torch.manual_seed(config["training"]["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = ConfocalGAPUNet(**{k: v for k, v in settings.items() if k != "name"}).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    validation = fixed_validation(counts["validation"], config, patches=2 if smoke else None)
    state = _restore_or_initialize(
        directory, model, optimizer, config, settings, validation, device
    )
    LOGGER.info(
        "%s: %s parameters; physical batch=%s accumulation=%s; NLL=%.6f",
        settings["name"],
        sum(p.numel() for p in model.parameters()),
        config["training"]["batch_size"],
        config["training"]["accumulation_steps"],
        state["curve"][-1]["photon_nll"],
    )
    _train_schedule(
        directory,
        model,
        optimizer,
        counts["train"],
        validation,
        config,
        settings,
        state,
        device,
        smoke,
    )
    result = _result(config, settings, model, state, smoke)
    _write_json(directory / "result.json", result)
    LOGGER.info("Saved %s", directory.resolve())
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def snapshot_sources(config):
    """Preserve the exact current implementation before an unattended long run."""
    destination = Path(config["output"]).parent / "source_snapshot"
    hashes = _source_hashes()
    for name, expected in hashes.items():
        content = Path(name).read_bytes()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            message = "A different implementation already owns this output directory"
            raise ValueError(message)
        target.write_bytes(content)
    _write_json(destination.parent / "source_snapshot.json", hashes)


def _run_requested(config, settings, counts, manifests, device, *, smoke, resume):
    if smoke:
        return [run_model(config, s, counts, manifests, device, smoke=True) for s in settings]
    directory = Path(config["output"]) / "pilot"
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "worker.lock"), timeout=0):
        status = {
            "pid": os.getpid(),
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(directory / "process.json", status)
        complete = False
        try:
            snapshot_sources(config)
            results = [
                run_model(config, s, counts, manifests, device, smoke=False, resume=resume)
                for s in settings
            ]
            _write_json(directory / "comparison.json", {r["model"]: r for r in results})
            complete = True
            return results
        finally:
            status.update(
                status="complete" if complete else "interrupted",
                updated_utc=datetime.now(timezone.utc).isoformat(),
            )
            _write_json(directory / "process.json", status)


def main():
    """Prepare real counts and explicitly select a short smoke test or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/confocal_gap4d_features32_batch4.yml")
    parser.add_argument("--model", choices=("all", "conv4d", "conv3d"), default="all")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--smoke-test", action="store_true")
    action.add_argument("--train", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Keep this task on physical GPU 0, independently of another caller's mapping.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if args.device == "cuda:0" else ""
    config = load_config(args.config)
    selected = [s for s in config["models"] if args.model in ("all", s["name"])]
    if not selected:
        parser.error("The requested model is not present in this configuration")
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and "allocator_fraction" in config["training"]:
        torch.cuda.set_per_process_memory_fraction(config["training"]["allocator_fraction"], device)
    counts, manifests = prepare_data(config)
    if args.prepare_only:
        return
    _run_requested(
        config, selected, counts, manifests, device, smoke=args.smoke_test, resume=args.resume
    )


if __name__ == "__main__":
    main()
