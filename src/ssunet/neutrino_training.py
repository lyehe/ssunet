"""Resumable fixed-budget photon training with streamed frozen evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from .datasets.neutrino import EpochPhotonBatchSampler
from .neutrino_evaluation import photon_score_summary
from .neutrino_streaming import stream_scores
from .photon_reference import photon_count_nll

LOGGER = logging.getLogger(__name__)


def _write_json(path, value):
    """Atomically replace a run-owned progress artifact."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_torch(path, value):
    """Publish complete checkpoints without exposing a partial pickle."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _step(model, optimizer, batch, device, settings):
    """Apply the unchanged count-weighted objective to observed A/B splits."""
    model.train()
    counts, target = batch["input"].to(device), batch["target"].to(device)
    live = batch["live"].to(device)
    optimizer.zero_grad(set_to_none=True)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with context:
        logits = model(counts, batch["positions_m"].to(device), live, batch["p"].to(device))
        loss = photon_count_nll(logits, target, live)
    if not bool(torch.isfinite(loss)):
        message = "Non-finite training likelihood"
        raise RuntimeError(message)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return float(loss.detach()) * len(counts), int(target.sum()), len(counts)


def _peak_rss():
    """Return a native Windows peak-working-set bound, or current RSS elsewhere."""
    process = psutil.Process()
    total = 0
    for owner in [process, *process.children(recursive=True)]:
        if owner.is_running():
            info = owner.memory_info()
            total += getattr(info, "peak_wset", info.rss)
    return total / 2**30


def _curve(model, validation, diagnostic, device, batch_size, epoch, step, interval):
    """Evaluate only the frozen development views at registered checkpoints."""
    records, values = stream_scores(validation, batch_size=batch_size, device=device, model=model)
    summary = photon_score_summary(values["model"], records)
    train_records, train_values = stream_scores(
        diagnostic, batch_size=batch_size, device=device, model=model
    )
    row = {
        "epoch": epoch,
        "optimizer_step": step,
        "validation_nats_per_photon": summary["all"]["nats_per_photon"],
        "primary_validation_nats_per_photon": summary["retained_1_to_10"]["nats_per_photon"],
        "zero_input_validation_nats_per_photon": summary["retained_0"]["nats_per_photon"],
        "fixed_training_nats_per_photon": float(
            train_values["model"].sum() / train_records.target_totals.sum()
        ),
        "fresh_training_nats_per_photon": interval[0] / interval[1] if interval[1] else None,
        "fresh_training_photons": int(interval[1]),
        "fresh_training_events": int(interval[2]),
    }
    return row, records, values["model"]


def _checkpoint(model, optimizer, state, config, identity, seed, device):
    """Keep cursor/RNG/optimizer state needed to resume the same shuffled epoch."""
    return {
        "schema_version": 2,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": state,
        "config": config,
        "identity": identity,
        "model_seed": seed,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _claim_run(directory):
    """Reject another verified live process; a stale PID alone is not a live job."""
    path = directory / "process.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior["status"] == "running" and psutil.pid_exists(prior["pid"]):
            process = psutil.Process(prior["pid"])
            if abs(process.create_time() - prior["process_created"]) < 0.01:
                message = f"This run is already owned by live process {prior['pid']}"
                raise RuntimeError(message)
    _write_json(
        path,
        {
            "pid": os.getpid(),
            "process_created": psutil.Process().create_time(),
            "status": "running",
        },
    )


def _new_state():
    """Start counters independent of wall-clock process lifetime."""
    return {
        "next_epoch": 0,
        "next_batch": 0,
        "optimizer_steps": 0,
        "training_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "checkpoint_seconds": 0.0,
        "training_examples": 0,
        "peak_process_tree_rss_gib": 0.0,
        "interval": [0.0, 0, 0],
        "curve": [],
        "best_primary": None,
        "best_epoch": None,
        "stopped_reason": None,
    }


def _save_curve(directory, state):
    """Write both machine-readable and ordinary spreadsheet learning records."""
    _write_json(directory / "learning_curve.json", state["curve"])
    with (directory / "learning_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(state["curve"][0]))
        writer.writeheader()
        writer.writerows(state["curve"])


def _evaluate_and_save(
    model, optimizer, state, config, identity, seed, device, views, directory, epoch
):
    """Select on primary validation only and retain every scheduled checkpoint."""
    begin = time.perf_counter()
    row, records, values = _curve(
        model,
        *views,
        device,
        config["training"]["batch_size"],
        epoch,
        state["optimizer_steps"],
        state["interval"],
    )
    state["evaluation_seconds"] += time.perf_counter() - begin
    state["curve"].append(row)
    state["interval"] = [0.0, 0, 0]
    primary = row["primary_validation_nats_per_photon"]
    if primary is None:
        message = "Scheduled primary validation has no target photons"
        raise RuntimeError(message)
    improved = state["best_primary"] is None or primary < state["best_primary"]
    if improved:
        state["best_primary"], state["best_epoch"] = primary, epoch
    payload = _checkpoint(model, optimizer, state, config, identity, seed, device)
    checkpoint_begin = time.perf_counter()
    _save_torch(directory / f"epoch_{epoch:03d}.pt", payload)
    if improved:
        _save_torch(directory / "best_low_count.pt", payload)
        np.savez_compressed(
            directory / "best_validation_scores.npz",
            event_ids=records.event_ids,
            configuration_ids=records.configuration_ids,
            retained_totals=records.retained_totals,
            target_totals=records.target_totals,
            model=values,
        )
    np.savez_compressed(
        directory / f"epoch_{epoch:03d}_scores.npz",
        event_ids=records.event_ids,
        configuration_ids=records.configuration_ids,
        retained_totals=records.retained_totals,
        target_totals=records.target_totals,
        model=values,
    )
    state["checkpoint_seconds"] += time.perf_counter() - checkpoint_begin
    _save_curve(directory, state)
    LOGGER.info(
        "seed=%s epoch=%s step=%s validation=%.5f low_count=%.5f train_seconds=%.1f",
        seed,
        epoch,
        state["optimizer_steps"],
        row["validation_nats_per_photon"],
        primary,
        state["training_seconds"],
    )


def _resource_stop(state, model, config, device):
    """Enforce the prospective update/evaluation and memory limits."""
    settings = config["training"]
    state["peak_process_tree_rss_gib"] = max(state["peak_process_tree_rss_gib"], _peak_rss())
    gpu = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
    if (
        state["peak_process_tree_rss_gib"] > settings["max_process_rss_gib"]
        or gpu > settings["max_gpu_allocated_gib"]
    ):
        return "memory_budget"
    if state["training_seconds"] >= settings["max_training_seconds_per_seed"]:
        return "training_time_budget"
    if state["evaluation_seconds"] >= settings["max_evaluation_seconds_per_seed"]:
        return "evaluation_time_budget"
    return None


def _run_epochs(
    model, optimizer, state, config, identity, seed, device, datasets, directory, stop_after_updates
):
    """Resume a deterministic sampler cursor without replaying optimized batches."""
    training, validation, diagnostic = datasets
    settings = config["training"]
    sampler = EpochPhotonBatchSampler(len(training), settings["batch_size"], seed=seed)
    workers = settings["num_workers"]
    loader = DataLoader(
        training,
        batch_sampler=sampler,
        num_workers=workers,
        generator=torch.Generator().manual_seed(seed),
        **({"persistent_workers": True, "multiprocessing_context": "spawn"} if workers else {}),
    )
    for epoch in range(state["next_epoch"], settings["epochs"]):
        sampler.epoch, sampler.start_batch = epoch, state["next_batch"]
        batch_cursor = state["next_batch"]
        iterator = iter(loader)
        while batch_cursor < (len(training) + settings["batch_size"] - 1) // settings["batch_size"]:
            stop = (
                _resource_stop(state, model, config, device)
                if state["optimizer_steps"] % 250 == 0
                else None
            )
            if state["training_seconds"] >= settings["max_training_seconds_per_seed"]:
                stop = "training_time_budget"
            if state["evaluation_seconds"] >= settings["max_evaluation_seconds_per_seed"]:
                stop = "evaluation_time_budget"
            if (directory.parent / "stop.request").exists():
                stop = "stop_requested"
            if stop_after_updates is not None and state["optimizer_steps"] >= stop_after_updates:
                stop = "requested_update_cap"
            if stop:
                state["stopped_reason"] = stop
                return
            begin = time.perf_counter()
            batch = next(iterator)
            contribution = _step(model, optimizer, batch, device, settings)
            state["training_seconds"] += time.perf_counter() - begin
            state["training_examples"] += contribution[2]
            state["interval"] = [
                a + b for a, b in zip(state["interval"], contribution, strict=True)
            ]
            batch_cursor += 1
            state["next_epoch"], state["next_batch"] = epoch, batch_cursor
            state["optimizer_steps"] += 1
            if state["optimizer_steps"] % 1000 == 0:
                _save_torch(
                    directory / "last.pt",
                    _checkpoint(model, optimizer, state, config, identity, seed, device),
                )
                _write_json(directory / "progress.json", state)
                LOGGER.info(
                    "seed=%s step=%s epoch=%.3f training_seconds=%.1f",
                    seed,
                    state["optimizer_steps"],
                    epoch + batch_cursor * settings["batch_size"] / len(training),
                    state["training_seconds"],
                )
        state["next_epoch"], state["next_batch"] = epoch + 1, 0
        if epoch + 1 in config["evaluation"]["epochs"] or epoch + 1 == settings["epochs"]:
            _evaluate_and_save(
                model,
                optimizer,
                state,
                config,
                identity,
                seed,
                device,
                (validation, diagnostic),
                directory,
                epoch + 1,
            )
        _save_torch(
            directory / "last.pt",
            _checkpoint(model, optimizer, state, config, identity, seed, device),
        )
    state["stopped_reason"] = "completed_fixed_epochs"


def train_photon_seed(
    config, seed, datasets, model_factory, directory, device, identity, *, stop_after_updates=None
):
    """Train/resume one seed with immutable identity and exact epoch/batch cursors."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "training_result.json").exists():
        result = json.loads((directory / "training_result.json").read_text())
        if result["identity"] != identity:
            message = "Completed run belongs to another data/configuration identity"
            raise ValueError(message)
        return result
    _claim_run(directory)
    device = torch.device(device)
    torch.manual_seed(seed)
    model = model_factory().to(device)
    settings = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    state = _new_state()
    if (directory / "last.pt").exists():
        payload = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        if payload["identity"] != identity or payload["model_seed"] != seed:
            message = "Resume checkpoint differs from the fixed data/configuration identity"
            raise ValueError(message)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        state = payload["training_state"]
        state["stopped_reason"] = None
        torch.set_rng_state(payload["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(payload["cuda_rng_state"], device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        if not state["curve"]:
            _evaluate_and_save(
                model, optimizer, state, config, identity, seed, device, datasets[1:], directory, 0
            )
        _run_epochs(
            model,
            optimizer,
            state,
            config,
            identity,
            seed,
            device,
            datasets,
            directory,
            stop_after_updates,
        )
        _save_torch(
            directory / "last.pt",
            _checkpoint(model, optimizer, state, config, identity, seed, device),
        )
        _write_json(directory / "progress.json", state)
        result = {
            "identity": identity,
            "seed": seed,
            "training_state": state,
            "parameters": sum(p.numel() for p in model.parameters()),
            "completed_fixed_epochs": state["stopped_reason"] == "completed_fixed_epochs",
            "process_segment_seconds": time.perf_counter() - started,
            "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else None,
            "peak_reserved_gpu_gib": torch.cuda.max_memory_reserved(device) / 2**30
            if device.type == "cuda"
            else None,
            "training_examples_per_second": state["training_examples"] / state["training_seconds"]
            if state["training_seconds"]
            else None,
        }
        if result["completed_fixed_epochs"]:
            _write_json(directory / "training_result.json", result)
        return result
    finally:
        _write_json(
            directory / "process.json",
            {
                "pid": os.getpid(),
                "process_created": psutil.Process().create_time(),
                "status": "stopped",
                "stopped_reason": state["stopped_reason"],
            },
        )


def configuration_identity(config, shards, code_hashes):
    """Bind a resumable run to its actual source code and development observations."""
    return {
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "train_sha256": shards["train"]["sha256"],
        "validation_sha256": shards["validation"]["sha256"],
        "code_sha256": code_hashes,
    }
