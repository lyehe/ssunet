"""Run a bounded, run-separated empirical marked-pulse prediction experiment."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml
from torch.utils.data import DataLoader

from ..ssunet.datasets.neutrino import EpochPhotonBatchSampler, shard_sha256
from ..ssunet.datasets.neutrino_pulses import MarkedPulseDataset
from ..ssunet.models.neutrino_pulses import DenseMarkedPulseReference
from ..ssunet.neutrino_pulse_baselines import MarkedPulseBaselines
from ..ssunet.neutrino_streaming import collect_training_reference, stream_scores
from ..ssunet.neutrino_training import _peak_rss, _save_torch, _step, _write_json
from ..ssunet.photon_reference import photon_count_log_probabilities
from .neutrino_prepare import write_json
from .neutrino_study import _atomic_scores, _code_hashes, _config_hash, _score_bundle

LOGGER = logging.getLogger(__name__)


def pulse_dataset(config, partition, *, training=False):
    """Read pulse multiplicities without interpreting fractional PE as photons."""
    return MarkedPulseDataset(
        Path(config["data"]["output"]) / f"{partition}.h5",
        partition,
        training=training,
        lazy=True,
        p=config["data"]["split_probability"],
        seed=config["data"]["photon_split_seed"],
    )


def _device(config, name):
    """Use a named device and bounded preprocessing threads."""
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = torch.device(name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def pulse_identity(config):
    """Freeze real observations, marked-pulse semantics and executable code."""
    root = Path(config["data"]["output"])
    code = _code_hashes()
    paths = [
        Path(__file__),
        Path(__file__).parents[1] / "ssunet/datasets/neutrino_pulses.py",
        Path(__file__).parents[1] / "ssunet/models/neutrino_pulses.py",
        Path(__file__).parents[1] / "ssunet/neutrino_pulse_baselines.py",
    ]
    code.update({str(path.resolve()): shard_sha256(path) for path in paths})
    return {
        "configuration_sha256": _config_hash(config),
        "code_sha256": code,
        "observation_sha256": {
            name: shard_sha256(root / f"{name}.h5") for name in ("train", "validation", "test")
        },
    }


def _training_reference(config):
    """Cache only O(geometry*marks*time) training counts and scalar event records."""
    root = Path(config["data"]["output"])
    training = pulse_dataset(config, "train")
    path = root / "training_reference.npz"
    if path.exists():
        metadata = json.loads((root / "training_reference.json").read_text())
        if metadata["observations_sha256"] != shard_sha256(training.path) or metadata[
            "reference_sha256"
        ] != shard_sha256(path):
            message = "The real-data training reference changed"
            raise ValueError(message)
        with np.load(path, allow_pickle=False) as source:
            reference = {key: source[key] for key in source.files}
    else:
        started = time.perf_counter()
        reference = collect_training_reference(training)
        np.savez_compressed(path, **reference)
        write_json(
            root / "training_reference.json",
            {
                "observations_sha256": shard_sha256(training.path),
                "reference_sha256": shard_sha256(path),
                "elapsed_seconds": time.perf_counter() - started,
                "count_object": "reconstructed_pulses_not_photoelectrons",
            },
        )
    return training, reference


def _fit_baseline(config, training, validation, reference, device):
    """Select the finite pulse-mixture grid using validation pulse log score only."""
    root = Path(config["data"]["output"])
    path = root / "baseline_selection.json"
    if path.exists():
        selected = json.loads(path.read_text())
        if selected["configuration_sha256"] != _config_hash(config):
            message = "Marked-pulse baseline belongs to another protocol"
            raise ValueError(message)
        return MarkedPulseBaselines(
            reference["observed_counts"], config["baselines"], device, selected=selected
        )
    bank = MarkedPulseBaselines(reference["observed_counts"], config["baselines"], device)
    parameters = [
        {"time_width_bins": width, "prior_fraction": fraction}
        for width in config["baselines"]["time_width_bins"]
        for fraction in config["baselines"]["prior_fractions"]
    ]
    totals = np.zeros(len(parameters))
    denominator = 0
    for batch in DataLoader(validation, batch_size=config["training"]["batch_size"]):
        a, b = batch["input"].to(device), batch["target"].to(device)
        denominator += int(b.sum())
        for index, parameter in enumerate(parameters):
            q = bank.mixture(a, parameter["time_width_bins"], parameter["prior_fraction"])
            totals[index] += float(-(b * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum())
    scores = totals / denominator
    best = int(np.argmin(scores))
    selected = {
        **parameters[best],
        "configuration_sha256": _config_hash(config),
        "candidates": [
            {**parameter, "all_validation_nats_per_marked_pulse": float(scores[index])}
            for index, parameter in enumerate(parameters)
        ],
        "test_used": False,
    }
    write_json(path, selected)
    return MarkedPulseBaselines(
        reference["observed_counts"], config["baselines"], device, selected=selected
    )


def _checkpoint(model, optimizer, state, identity, device):
    """Save the exact optimizer cursor and RNG state for a bounded real-data run."""
    return {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": state,
        "identity": identity,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _validation(model, dataset, config, device):
    """Select using all held-out marked pulses; a low-count real fold is unavailable."""
    records, values = stream_scores(
        dataset, batch_size=config["training"]["batch_size"], device=device, model=model
    )
    return float(values["model"].sum() / records.target_totals.sum()), records, values


def _train_one(config, identity, fitting, validation, directory, device, seed):  # noqa: C901
    """Resume the same marked-pulse optimization without replaying optimized batches."""
    directory.mkdir(exist_ok=True, parents=True)
    if (directory / "training_result.json").exists():
        result = json.loads((directory / "training_result.json").read_text())
        if result["identity"] != identity:
            message = "Completed real run has another observation/implementation identity"
            raise ValueError(message)
        return result
    process_path = directory / "process.json"
    if process_path.exists():
        prior = json.loads(process_path.read_text())
        if (
            prior["status"] == "running"
            and psutil.pid_exists(prior["pid"])
            and abs(psutil.Process(prior["pid"]).create_time() - prior["process_created"]) < 0.01
        ):
            message = "The marked-pulse run already has a verified live owner"
            raise RuntimeError(message)
    _write_json(
        process_path,
        {
            "pid": os.getpid(),
            "process_created": psutil.Process().create_time(),
            "status": "running",
        },
    )
    torch.manual_seed(seed)
    marks = len(config["data"]["charge_interior_edges_pe"]) + 1
    model = DenseMarkedPulseReference(marks, config["model"]["channels"]).to(device)
    settings = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    state = {
        "next_epoch": 0,
        "next_batch": 0,
        "optimizer_steps": 0,
        "training_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "training_examples": 0,
        "curve": [],
        "best_validation": None,
        "best_epoch": None,
        "peak_process_rss_gib": 0.0,
        "stopped_reason": None,
    }
    if (directory / "last.pt").exists():
        checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["identity"] != identity:
            message = "Real resume checkpoint differs from the frozen contract"
            raise ValueError(message)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        state = checkpoint["training_state"]
        state["stopped_reason"] = None
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
    sampler = EpochPhotonBatchSampler(len(fitting), settings["batch_size"], seed=seed)
    loader = DataLoader(
        fitting, batch_sampler=sampler, num_workers=0, generator=torch.Generator().manual_seed(seed)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(state["next_epoch"], settings["epochs"]):
            sampler.epoch, sampler.start_batch = epoch, state["next_batch"]
            iterator = iter(loader)
            cursor = state["next_batch"]
            batches = (len(fitting) + settings["batch_size"] - 1) // settings["batch_size"]
            while cursor < batches:
                if state["optimizer_steps"] % 100 == 0:
                    state["peak_process_rss_gib"] = max(state["peak_process_rss_gib"], _peak_rss())
                    gpu = (
                        torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == "cuda"
                        else 0
                    )
                    if (
                        state["peak_process_rss_gib"] > settings["max_process_rss_gib"]
                        or gpu > settings["max_gpu_allocated_gib"]
                    ):
                        state["stopped_reason"] = "memory_budget"
                if state["training_seconds"] >= settings["max_training_seconds_per_seed"]:
                    state["stopped_reason"] = "training_time_budget"
                if state["evaluation_seconds"] >= settings["max_evaluation_seconds_per_seed"]:
                    state["stopped_reason"] = "evaluation_time_budget"
                if state["stopped_reason"]:
                    break
                started = time.perf_counter()
                batch = next(iterator)
                _step(model, optimizer, batch, device, settings)
                state["training_seconds"] += time.perf_counter() - started
                state["training_examples"] += len(batch["input"])
                cursor += 1
                state["next_epoch"], state["next_batch"] = epoch, cursor
                state["optimizer_steps"] += 1
                if state["optimizer_steps"] % 250 == 0:
                    _save_torch(
                        directory / "last.pt",
                        _checkpoint(model, optimizer, state, identity, device),
                    )
                    _write_json(directory / "progress.json", state)
                    LOGGER.info(
                        "Pulse seed=%s step=%s train_seconds=%.1f",
                        seed,
                        state["optimizer_steps"],
                        state["training_seconds"],
                    )
            if state["stopped_reason"]:
                break
            state["next_epoch"], state["next_batch"] = epoch + 1, 0
            started = time.perf_counter()
            score, records, values = _validation(model, validation, config, device)
            state["evaluation_seconds"] += time.perf_counter() - started
            row = {
                "epoch": epoch + 1,
                "optimizer_steps": state["optimizer_steps"],
                "all_validation_nats_per_marked_pulse": score,
            }
            state["curve"].append(row)
            if state["best_validation"] is None or score < state["best_validation"]:
                state["best_validation"], state["best_epoch"] = score, epoch + 1
                _save_torch(
                    directory / "best.pt", _checkpoint(model, optimizer, state, identity, device)
                )
                _atomic_scores(directory / "best_validation_scores.npz", records, values)
            _save_torch(
                directory / "last.pt", _checkpoint(model, optimizer, state, identity, device)
            )
            _write_json(directory / "learning_curve.json", state["curve"])
            LOGGER.info("Pulse seed=%s epoch=%s marked_validation=%.5f", seed, epoch + 1, score)
        if state["next_epoch"] == settings["epochs"]:
            state["stopped_reason"] = "completed_fixed_epochs"
        _save_torch(directory / "last.pt", _checkpoint(model, optimizer, state, identity, device))
        _write_json(directory / "progress.json", state)
        result = {
            "identity": identity,
            "seed": seed,
            "state": state,
            "completed_fixed_epochs": state["stopped_reason"] == "completed_fixed_epochs",
            "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else None,
            "training_examples_per_second": state["training_examples"] / state["training_seconds"]
            if state["training_seconds"]
            else None,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        if result["completed_fixed_epochs"]:
            _write_json(directory / "training_result.json", result)
        return result
    finally:
        _write_json(
            process_path,
            {
                "pid": os.getpid(),
                "process_created": psutil.Process().create_time(),
                "status": "stopped",
                "reason": state["stopped_reason"],
            },
        )


def train_pulses(config, device):
    """Fit on one real run and select using a different run without test access."""
    root = Path(config["data"]["output"])
    device = _device(config, device)
    identity = pulse_identity(config)
    manifest = root / "training_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
        message = "The empirical pulse study changed after its training freeze"
        raise ValueError(message)
    write_json(manifest, {"identity": identity, "configuration": config, "test_scored": False})
    training, reference = _training_reference(config)
    validation, fitting = (
        pulse_dataset(config, "validation"),
        pulse_dataset(config, "train", training=True),
    )
    bank = _fit_baseline(config, training, validation, reference, device)
    records, baseline = stream_scores(
        validation, batch_size=config["training"]["batch_size"], device=device, baselines=bank
    )
    _atomic_scores(root / "validation_baseline_scores.npz", records, baseline)
    runs = []
    for seed in config["training"]["seeds"]:
        result = _train_one(
            config, identity, fitting, validation, root / f"runs/seed_{seed}", device, seed
        )
        runs.append(result)
        if not result["completed_fixed_epochs"]:
            write_json(root / "development_progress.json", {"runs": runs})
            return
    lock = {
        "identity": identity,
        "runs": runs,
        "checkpoints": {
            str(seed): {
                "path": str(root / f"runs/seed_{seed}/best.pt"),
                "sha256": shard_sha256(root / f"runs/seed_{seed}/best.pt"),
            }
            for seed in config["training"]["seeds"]
        },
        "baseline_sha256": shard_sha256(root / "baseline_selection.json"),
        "source_contract": config["contract"],
    }
    if not (root / "analysis_lock.json").exists():
        write_json(root / "analysis_lock.json", lock)
    for dataset in (training, validation, fitting):
        dataset.close()


def profile_pulses(config, device):
    """Measure full spatial/mark support on training observations before scheduling."""
    root = Path(config["data"]["output"])
    destination = root / "engineering_profile.json"
    if destination.exists():
        message = "A real-data engineering profile already exists"
        raise FileExistsError(message)
    device = _device(config, device)
    dataset = pulse_dataset(config, "train", training=True)
    torch.manual_seed(42)
    model = DenseMarkedPulseReference(dataset.charge_categories, config["model"]["channels"]).to(
        device
    )
    settings = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    loader = iter(DataLoader(dataset, batch_size=settings["batch_size"], shuffle=False))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started, measured, events = None, 0.0, 0
    for step in range(25):
        if step == 5:
            started = time.perf_counter()
        batch = next(loader)
        _step(model, optimizer, batch, device, settings)
        if step >= 5:
            events += len(batch["input"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured = time.perf_counter() - started
    write_json(
        destination,
        {
            "scope": "training_only_engineering; 20_measured_updates_plus_5_warmup",
            "identity": pulse_identity(config),
            "examples": events,
            "elapsed_seconds": measured,
            "examples_per_second": events / measured,
            "projected_fixed_epoch_training_seconds": len(dataset)
            * settings["epochs"]
            * measured
            / events,
            "peak_process_rss_gib": _peak_rss(),
            "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else None,
            "precision": "full_float32",
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device),
            "GPU_may_be_shared_with_mechanism_runs": True,
            "validation_or_test_used": False,
        },
    )
    dataset.close()


def _pulse_units(value):
    """Label generic count-score records with their actual observation unit."""
    if isinstance(value, dict):
        return {
            key.replace("photons", "pulses").replace("photon", "pulse"): _pulse_units(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_pulse_units(item) for item in value]
    return value


@torch.inference_mode()
def marginal_arrival_scores(model, testing, config, device):
    """Score the arrival marginal after summing over the observed charge categories."""
    values = []
    model.eval()
    marks = len(config["data"]["charge_interior_edges_pe"]) + 1
    for batch in DataLoader(testing, batch_size=config["training"]["batch_size"]):
        a, b = batch["input"].to(device), batch["target"].to(device)
        logits = model(
            a, batch["positions_m"].to(device), batch["live"].to(device), batch["p"].to(device)
        )
        q = photon_count_log_probabilities(logits, batch["live"].to(device)).exp()
        q = q.reshape(len(a), -1, marks, a.shape[-1]).sum(2)
        target = b.reshape(len(a), -1, marks, a.shape[-1]).sum(2)
        values.extend(
            (-(target * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum((1, 2))).cpu().tolist()
        )
    return np.asarray(values)


def test_pulses(config, device):
    """Score the sealed third real run once with explicit domain-shift limitations."""
    root = Path(config["data"]["output"])
    if (root / "test_results.json").exists():
        message = "The run-separated real-data test already has results"
        raise FileExistsError(message)
    lock = json.loads((root / "analysis_lock.json").read_text())
    if lock["identity"] != pulse_identity(config) or lock["baseline_sha256"] != shard_sha256(
        root / "baseline_selection.json"
    ):
        message = "The real-data analysis changed after model selection"
        raise ValueError(message)
    write_json(
        root / "test_opened.json",
        {
            "analysis_lock_sha256": shard_sha256(root / "analysis_lock.json"),
            "scope": "one_predeclared_real_run; empirical_marked_pulse_prediction_only",
        },
    )
    device = _device(config, device)
    training, reference = _training_reference(config)
    testing = pulse_dataset(config, "test")
    selected = json.loads((root / "baseline_selection.json").read_text())
    bank = MarkedPulseBaselines(
        reference["observed_counts"], config["baselines"], device, selected=selected
    )
    values, arrivals = [], []
    for seed in config["training"]["seeds"]:
        checkpoint = lock["checkpoints"][str(seed)]
        if shard_sha256(checkpoint["path"]) != checkpoint["sha256"]:
            message = "A real-data selected checkpoint changed"
            raise ValueError(message)
        model = DenseMarkedPulseReference(
            len(config["data"]["charge_interior_edges_pe"]) + 1, config["model"]["channels"]
        ).to(device)
        model.load_state_dict(
            torch.load(checkpoint["path"], map_location="cpu", weights_only=True)["state_dict"]
        )
        records, scored = stream_scores(
            testing,
            batch_size=config["training"]["batch_size"],
            device=device,
            model=model,
            baselines=bank,
        )
        _atomic_scores(root / f"test_seed_{seed}.npz", records, scored)
        values.append(scored)
        arrivals.append(marginal_arrival_scores(model, testing, config, device))
    means = {key: np.mean([v[key] for v in values], axis=0) for key in values[0]}
    score_config = {
        "evaluation": {
            "bootstrap_seed": config["evaluation"]["bootstrap_seed"],
            "bootstrap_draws": config["evaluation"]["bootstrap_draws"],
            "bootstrap_chunk_draws": 64,
        }
    }
    result = _pulse_units(_score_bundle(records, means, score_config, bank.winner))
    result.update(
        {
            "scope": (
                "run_separated_real_marked_pulse_prediction; fixed_spatial_cells; "
                "coarse_1us_bins; no_photon_GT"
            ),
            "arrival_marginal_model_nats_per_pulse": float(
                np.mean(arrivals, axis=0).sum() / records.target_totals.sum()
            ),
            "confidence_scope": config["evaluation"]["confidence_scope"],
            "number_of_independent_test_runs": 1,
            "no_physical_no_event_inference": True,
            "live_status_or_DAQ_exposure_not_provided": True,
            "real_low_count_test_available": bool(records.strata["retained_1_to_10"].any()),
            "domain_shifts": [
                "2010_IC59_to_2013_IC86",
                "different_trigger_and_filter_selections",
                "reported_position_drift_and_spatial_quantization",
                "unknown_clock_tail_and_live_status",
            ],
            "test_event_count_not_multiplied_by_model_seeds": len(records.event_ids),
        }
    )
    _atomic_scores(
        root / "test_mean_seed_scores.npz", records, means, arrival_model=np.mean(arrivals, axis=0)
    )
    write_json(root / "test_results.json", result)
    testing.close()
    training.close()


def main():
    """Run a frozen empirical real-data training or first test stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("profile", "train", "test"))
    parser.add_argument("--config", type=Path, default=Path("config/neutrino_pulses.yml"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = yaml.safe_load(args.config.read_text())
    if args.action == "profile":
        profile_pulses(config, args.device)
    elif args.action == "train":
        train_pulses(config, args.device)
    else:
        test_pulses(config, args.device)


if __name__ == "__main__":
    main()
