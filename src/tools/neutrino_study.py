"""Run bounded engineering, development and locked-test stages of the photon study."""

from __future__ import annotations

import argparse
import hashlib
import inspect
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

if __package__ == "src.tools":
    from ..ssunet.datasets.neutrino import (
        EpochPhotonBatchSampler,
        NeutrinoPhotonDataset,
        shard_sha256,
    )
    from ..ssunet.distributions import logits_to_log_probabilities
    from ..ssunet.models.neutrino_dense import DensePhotonReference
    from ..ssunet.neutrino_evaluation import (
        PhotonScoreRecords,
        paired_photon_interval,
        photon_score_summary,
    )
    from ..ssunet.neutrino_streaming import (
        BoundedPhotonDonors,
        PhotonBaselineBank,
        collect_training_reference,
        select_baselines,
        stream_donor_scores,
        stream_scores,
    )
    from ..ssunet.neutrino_training import configuration_identity, train_photon_seed
    from ..ssunet.photon_reference import photon_count_log_probabilities, photon_count_nll
    from .neutrino_pilot import _training_step
    from .neutrino_prepare import write_json
else:
    from ssunet.datasets.neutrino import (
        EpochPhotonBatchSampler,
        NeutrinoPhotonDataset,
        shard_sha256,
    )
    from ssunet.distributions import logits_to_log_probabilities
    from ssunet.models.neutrino_dense import DensePhotonReference
    from ssunet.neutrino_evaluation import (
        PhotonScoreRecords,
        paired_photon_interval,
        photon_score_summary,
    )
    from ssunet.neutrino_streaming import (
        BoundedPhotonDonors,
        PhotonBaselineBank,
        collect_training_reference,
        select_baselines,
        stream_donor_scores,
        stream_scores,
    )
    from ssunet.neutrino_training import configuration_identity, train_photon_seed
    from ssunet.photon_reference import photon_count_log_probabilities, photon_count_nll
    from tools.neutrino_pilot import _training_step
    from tools.neutrino_prepare import write_json

LOGGER = logging.getLogger(__name__)


def _root(config):
    """Return the configured study artifact directory."""
    return Path(config["data"]["output"])


def _config_hash(config):
    """Fingerprint the prospective configuration with deterministic serialization."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _device(config, name):
    """Select a backend explicitly and keep CPU preprocessing threads bounded."""
    device = torch.device(name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(config["training"]["cpu_threads"])
    return device


def _dataset(config, partition, *, engineering=False, training=False, p=None, ids=None):
    """Open one named fold lazily; fitting never receives a test catalogue view."""
    prefix = "engineering_" if engineering else ""
    return NeutrinoPhotonDataset(
        _root(config) / f"{prefix}{partition}.h5",
        partition,
        lazy=True,
        training=training,
        seed=config["data"]["photon_split_seed"],
        p=config["data"]["split_probability"] if p is None else p,
        selected_event_ids=ids,
    )


def _rss_gib():
    """Measure process plus live worker resident memory, excluding OS disk cache."""
    process = psutil.Process()
    children = process.children(recursive=True)
    info = process.memory_info()
    total = getattr(info, "peak_wset", info.rss)
    for child in children:
        if child.is_running():
            child_info = child.memory_info()
            total += getattr(child_info, "peak_wset", child_info.rss)
    return total / 2**30


def _code_hashes():
    """Record code that affects loading, likelihoods, optimization and selection."""
    objects = (
        NeutrinoPhotonDataset,
        DensePhotonReference,
        stream_scores,
        paired_photon_interval,
        _training_step,
    )
    objects += (photon_count_log_probabilities, photon_count_nll, logits_to_log_probabilities)
    objects += (train_photon_seed,)
    paths = {Path(inspect.getfile(inspect.unwrap(obj))).resolve() for obj in objects}
    paths.add(Path(__file__).resolve())
    return {str(path): shard_sha256(path) for path in sorted(paths)}


def _assert_source_and_shards(config, names):
    """Verify the declared source interpretation and requested observation identities."""
    root = _root(config)
    audit = json.loads((root / "source_audit.json").read_text())
    if audit["interpretation"] != config["source"]["allowed_contract"]:
        message = "Source contract differs from the prospective study configuration"
        raise ValueError(message)
    shards = json.loads((root / "shards.json").read_text())
    for name in names:
        if shard_sha256(shards[name]["path"]) != shards[name]["sha256"]:
            message = f"Observation shard identity changed: {name}"
            raise ValueError(message)
    return shards


def engineering_profile(config, device_name, *, tag="engineering_profile"):
    """Measure 500 optimizer updates on engineering TRAINING observations only."""
    root = _root(config)
    destination = root / f"{tag}.json"
    if destination.exists():
        message = "Engineering profile exists; preserve it and use a distinct profile tag"
        raise FileExistsError(message)
    _assert_source_and_shards(config, ["engineering_train"])
    device = _device(config, device_name)
    dataset = _dataset(config, "train", engineering=True, training=True)
    sampler = EpochPhotonBatchSampler(len(dataset), config["training"]["batch_size"], seed=42)
    workers = config["training"]["num_workers"]
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        **({"persistent_workers": True, "multiprocessing_context": "spawn"} if workers else {}),
    )
    torch.manual_seed(42)
    model = DensePhotonReference(config["model"]["channels"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    iterator, events = iter(loader), 0
    peak_rss = _rss_gib()
    start = None
    data_seconds = update_seconds = 0.0
    for step in range(510):
        if step == 10:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
        before_data = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            sampler.epoch += 1
            iterator = iter(loader)
            batch = next(iterator)
        before_update = time.perf_counter()
        _training_step(model, optimizer, batch, device, config["training"])
        after_update = time.perf_counter()
        if step % 50 == 0 or step == 509:
            peak_rss = max(peak_rss, _rss_gib())
        if step >= 10:
            events += len(batch["input"])
            data_seconds += before_update - before_data
            update_seconds += after_update - before_update
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    gpu = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    result = {
        "scope": "engineering_training_only_no_validation_or_test_scores",
        "config_sha256": _config_hash(config),
        "code_sha256": _code_hashes(),
        "optimizer_steps": 500,
        "warmup_steps": 10,
        "events": events,
        "elapsed_seconds": elapsed,
        "training_events_per_second": events / elapsed,
        "num_workers": workers,
        "data_loading_seconds": data_seconds,
        "optimizer_update_seconds": update_seconds,
        "peak_process_tree_rss_gib": peak_rss,
        "peak_allocated_gpu_gib": gpu,
        "peak_reserved_gpu_gib": torch.cuda.max_memory_reserved(device) / 2**30
        if device.type == "cuda"
        else None,
        "device_used_gpu_gib": (
            torch.cuda.mem_get_info(device)[1] - torch.cuda.mem_get_info(device)[0]
        )
        / 2**30
        if device.type == "cuda"
        else None,
        "memory_target_met": peak_rss < config["training"]["max_process_rss_gib"]
        and (gpu is None or gpu < config["training"]["max_gpu_allocated_gib"]),
        "all_gradients_and_losses_finite": True,
        "validation_scored": False,
        "test_scored": False,
        "estimated_ten_epoch_update_seconds": 10
        * json.loads((root / "shards.json").read_text())["train"]["events"]
        / (events / elapsed),
    }
    write_json(destination, result)
    dataset.close()
    LOGGER.info("Engineering profile: %s", json.dumps(result))
    return result


def _training_reference(config):
    """Fit or verify a reusable training-only prior/donor scalar index."""
    root = _root(config)
    destination = root / "training_reference.npz"
    training = _dataset(config, "train")
    metadata = root / "training_reference.json"
    if destination.exists():
        state = json.loads(metadata.read_text())
        if state["config_sha256"] != _config_hash(config):
            message = "Training reference belongs to another study configuration"
            raise ValueError(message)
        if shard_sha256(destination) != state["sha256"]:
            message = "Training prior/donor scalar reference changed"
            raise ValueError(message)
        with np.load(destination, allow_pickle=False) as handle:
            values = {key: handle[key] for key in handle.files}
        return training, values
    begin = time.perf_counter()
    values = collect_training_reference(training)
    np.savez_compressed(destination, **values)
    write_json(
        metadata,
        {
            "config_sha256": _config_hash(config),
            "sha256": shard_sha256(destination),
            "training_shard_sha256": shard_sha256(training.path),
            "fit_seconds": time.perf_counter() - begin,
            "events": len(training),
            "photons_in_window": values["photons_in_window"],
            "outside_window_photons": values["outside_window_photons"],
            "empty_observations": values["empty_observations"],
            "physics_truth_used": False,
        },
    )
    return training, values


def fit_study_baselines(config, device_name):
    """Fit the registered finite baseline grid on development data, never test."""
    root = _root(config)
    if (root / "baseline_selection.json").exists():
        message = "Baseline selection is already recorded and cannot be silently retuned"
        raise FileExistsError(message)
    profile = json.loads((root / "engineering_profile.json").read_text())
    if not profile["memory_target_met"]:
        message = "Engineering memory gate failed before baseline/model fitting"
        raise RuntimeError(message)
    _assert_source_and_shards(config, ["train", "validation"])
    device = _device(config, device_name)
    training, reference = _training_reference(config)
    validation = _dataset(config, "validation")
    bank = PhotonBaselineBank(
        reference["observed_counts"],
        training.positions_m,
        training.live,
        training.edges_ns,
        config["baselines"],
        device,
    )
    control_code_sha256 = _code_hashes()
    begin = time.perf_counter()
    LOGGER.info("Fitting registered baseline candidates on validation; test remains reserved")
    selection = select_baselines(reference, validation, bank, config["training"]["batch_size"])
    selection.update(
        {
            "config_sha256": _config_hash(config),
            "elapsed_seconds": time.perf_counter() - begin,
            "test_scored": False,
            "control_code_sha256": control_code_sha256,
        }
    )
    write_json(root / "baseline_selection.json", selection)
    records, scores = stream_scores(
        validation, batch_size=config["training"]["batch_size"], device=device, baselines=bank
    )
    np.savez_compressed(
        root / "validation_baseline_scores.npz",
        event_ids=records.event_ids,
        configuration_ids=records.configuration_ids,
        retained_totals=records.retained_totals,
        target_totals=records.target_totals,
        **scores,
    )
    write_json(
        root / "validation_baseline_scores.json",
        {name: photon_score_summary(value, records) for name, value in scores.items()},
    )
    training.close()
    validation.close()
    LOGGER.info(
        "Selected baseline %s, low-count NLL %.6f",
        bank.winner,
        selection["method_primary_nll"][bank.winner],
    )
    return selection


def _read_scalar_scores(path):
    """Read only O(events) likelihood/denominator records."""
    with np.load(path, allow_pickle=False) as source:
        records = PhotonScoreRecords(
            source["event_ids"],
            source["configuration_ids"],
            source["retained_totals"],
            source["target_totals"],
        )
        metadata = {
            "event_ids",
            "configuration_ids",
            "retained_totals",
            "target_totals",
            "exact_donor",
            "donor_event_ids",
        }
        values = {key: source[key] for key in source.files if key not in metadata}
    return records, values


def _assert_same_records(left, right):
    """Require identical frozen observations before comparing two sets of losses."""
    for attribute in ("event_ids", "configuration_ids", "retained_totals", "target_totals"):
        if not np.array_equal(getattr(left, attribute), getattr(right, attribute)):
            message = "Frozen scalar records differ between model and baseline scoring"
            raise ValueError(message)


def _bank_from_selection(config, training, reference, device):
    """Reconstruct fixed training-derived priors and frozen validation choices."""
    selected = json.loads((_root(config) / "baseline_selection.json").read_text())
    if selected["config_sha256"] != _config_hash(config):
        message = "Baseline selection belongs to another prospective configuration"
        raise ValueError(message)
    bank = PhotonBaselineBank(
        reference["observed_counts"],
        training.positions_m,
        training.live,
        training.edges_ns,
        config["baselines"],
        device,
    )
    bank.selected_prior_parameters = selected["selected_prior"]
    bank.selected_mixture_parameters = selected["selected_mixture"]
    bank.winner = selected["selected_method"]
    bank._set_priors()
    return bank


def _score_bundle(records, values, config, baseline_name, exact=None):
    """Aggregate photon-weighted scores and paired configuration-level differences."""
    masks = records.strata
    contrasts = [baseline_name]
    if "matched_donor" in values:
        contrasts.append("matched_donor")
    result = {
        "scores": {name: photon_score_summary(losses, records) for name, losses in values.items()},
        "contrasts": {},
    }
    for stratum, mask in masks.items():
        result["contrasts"][stratum] = {
            name: (
                paired_photon_interval(
                    values["model"] - values[name],
                    records,
                    mask,
                    seed=config["evaluation"]["bootstrap_seed"],
                    draws=config["evaluation"]["bootstrap_draws"],
                    chunk_size=config["evaluation"]["bootstrap_chunk_draws"],
                )
                if name != "matched_donor" or exact is None or exact[mask].all()
                else None
            )
            for name in contrasts
        }
    if exact is not None:
        result["exact_donor_fraction"] = {
            name: float(exact[mask].mean()) if mask.any() else None for name, mask in masks.items()
        }
        result["donor_interpretation"] = "contrasts_require_exact_matching_across_the_stratum"
    return result


def _selected_validation_result(
    config,
    result,
    directory,
    training,
    validation,
    reference,
    device,
    baseline_records,
    baseline_scores,
):
    """Score the selected model and count-matched controls without using test observations."""
    destination = directory / "selected_validation_result.json"
    if destination.exists():
        return json.loads(destination.read_text())
    begin = time.perf_counter()
    model = DensePhotonReference(config["model"]["channels"]).to(device)
    checkpoint = torch.load(directory / "best_low_count.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    records, own = _read_scalar_scores(directory / "best_validation_scores.npz")
    _assert_same_records(records, baseline_records)
    donors = BoundedPhotonDonors(
        training,
        reference["retained_totals"],
        seed=config["evaluation"]["donor_seed"],
        cache_events=config["evaluation"]["donor_cache_events"],
    )
    donor_nll, exact, donor_ids = stream_donor_scores(
        model,
        validation,
        donors,
        batch_size=config["training"]["batch_size"],
        device=device,
        draws=config["evaluation"]["donor_draws"],
    )
    values = {**baseline_scores, **own, "matched_donor": donor_nll}
    primary = records.strata["retained_1_to_10"]
    if not exact[primary].all():
        message = "The primary donor control lacks exact retained-count matches"
        raise RuntimeError(message)
    bank = _bank_from_selection(config, training, reference, device)
    bundle = _score_bundle(records, values, config, bank.winner, exact)
    post_seconds = time.perf_counter() - begin
    np.savez_compressed(
        directory / "selected_validation_evidence.npz",
        event_ids=records.event_ids,
        configuration_ids=records.configuration_ids,
        retained_totals=records.retained_totals,
        target_totals=records.target_totals,
        exact_donor=exact,
        donor_event_ids=donor_ids,
        **values,
    )
    bundle.update(
        {
            "seed": result["seed"],
            "selected_epoch": checkpoint["training_state"]["best_epoch"],
            "baseline": bank.winner,
            "post_training_evaluation_seconds": post_seconds,
            "total_evaluation_seconds": result["training_state"]["evaluation_seconds"]
            + post_seconds,
            "test_scored": False,
        }
    )
    write_json(destination, bundle)
    return bundle


def _aggregate_evidence(config, records, paths, baseline_name):
    """Average seed losses on the same events without multiplying the sample size."""
    collected, matching = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            candidate = PhotonScoreRecords(
                source["event_ids"],
                source["configuration_ids"],
                source["retained_totals"],
                source["target_totals"],
            )
            _assert_same_records(records, candidate)
            collected.append(
                {name: source[name] for name in ("model", baseline_name, "matched_donor")}
            )
            matching.append(source["exact_donor"])
    means = {name: np.mean([row[name] for row in collected], axis=0) for name in collected[0]}
    report = _score_bundle(records, means, config, baseline_name, np.logical_and.reduce(matching))
    report["model_seed_scores"] = [photon_score_summary(row["model"], records) for row in collected]
    return report


def train_study(config, device_name):
    """Run or resume all fixed seeds and assess the development gate before test lock."""
    root = _root(config)
    shards = _assert_source_and_shards(config, ["train", "validation"])
    device = _device(config, device_name)
    training, reference = _training_reference(config)
    validation = _dataset(config, "validation")
    diagnostic_ids = training.event_ids[training.indices][
        np.random.default_rng(config["evaluation"]["diagnostic_seed"]).permutation(len(training))[
            : config["evaluation"]["fixed_training_diagnostic_events"]
        ]
    ]
    diagnostic = _dataset(config, "train", ids=diagnostic_ids)
    fitting = _dataset(config, "train", training=True)
    baseline_records, baseline_scores = _read_scalar_scores(root / "validation_baseline_scores.npz")
    bank = _bank_from_selection(config, training, reference, device)
    identity = configuration_identity(config, shards, _code_hashes())
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    write_json(
        root / "training_manifest.json",
        {
            "identity": identity,
            "baseline": bank.winner,
            "test_scored": False,
            "device": str(device),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
        },
    )
    results, selected = [], []
    for seed in config["training"]["seeds"]:
        directory = runs / f"seed_{seed}"
        result = train_photon_seed(
            config,
            seed,
            (fitting, validation, diagnostic),
            lambda: DensePhotonReference(config["model"]["channels"]),
            directory,
            device,
            identity,
        )
        results.append(result)
        if not result["completed_fixed_epochs"]:
            write_json(root / "development_progress.json", {"runs": results, "test_scored": False})
            return {
                "completed": False,
                "stopped_reason": result["training_state"]["stopped_reason"],
            }
        selected.append(
            _selected_validation_result(
                config,
                result,
                directory,
                training,
                validation,
                reference,
                device,
                baseline_records,
                baseline_scores,
            )
        )
    paths = [
        runs / f"seed_{seed}/selected_validation_evidence.npz"
        for seed in config["training"]["seeds"]
    ]
    aggregate = _aggregate_evidence(config, baseline_records, paths, bank.winner)
    primary = aggregate["contrasts"]["retained_1_to_10"]
    margin = config["evaluation"]["practical_improvement_nats_per_photon"]
    resource = all(row["completed_fixed_epochs"] for row in results) and all(
        row["total_evaluation_seconds"] <= config["training"]["max_evaluation_seconds_per_seed"]
        for row in selected
    )
    point_gate = (
        primary[bank.winner]["model_minus_control_nats_per_photon"] <= -margin
        and primary["matched_donor"]["model_minus_control_nats_per_photon"] < 0
    )
    summary = {
        "identity": identity,
        "runs": results,
        "selected_validation": selected,
        "aggregate": aggregate,
        "baseline": bank.winner,
        "resource_gate": resource,
        "primary_validation_point_gate": point_gate,
        "eligible_for_test_lock": point_gate and resource,
        "test_scored": False,
    }
    write_json(root / "development_summary.json", summary)
    for dataset in (training, validation, diagnostic, fitting):
        dataset.close()
    return summary


def lock_analysis(config):
    """Seal chosen development models/settings before a test observation is scored."""
    root = _root(config)
    destination = root / "analysis_lock.json"
    if destination.exists():
        message = "Analysis lock already exists and is immutable"
        raise FileExistsError(message)
    development = json.loads((root / "development_summary.json").read_text())
    if not development["eligible_for_test_lock"]:
        message = (
            "Development gate has not passed; retain test for development-independent evaluation"
        )
        raise RuntimeError(message)
    shards = _assert_source_and_shards(config, ["train", "validation", "test"])
    checkpoints = {}
    for seed in config["training"]["seeds"]:
        path = root / f"runs/seed_{seed}/best_low_count.pt"
        checkpoints[str(seed)] = {"path": str(path), "sha256": shard_sha256(path)}
    lock = {
        "schema_version": 1,
        "config": config,
        "config_sha256": _config_hash(config),
        "protocol_sha256": shard_sha256("plans/neutrino_study_protocol.txt"),
        "cohort_manifest_sha256": shard_sha256(root / "cohort_manifest.parquet"),
        "baseline_selection_sha256": shard_sha256(root / "baseline_selection.json"),
        "training_reference_sha256": shard_sha256(root / "training_reference.npz"),
        "training_reference_metadata_sha256": shard_sha256(root / "training_reference.json"),
        "development_summary_sha256": shard_sha256(root / "development_summary.json"),
        "code_sha256": _code_hashes(),
        "checkpoints": checkpoints,
        "observations": {
            name: {"path": shards[name]["path"], "sha256": shards[name]["sha256"]}
            for name in ("train", "validation", "test")
        },
        "baseline": development["baseline"],
        "created_unix_seconds": time.time(),
        "statistical_contract": config["source"]["allowed_contract"],
        "model_selection": config["evaluation"]["checkpoint_selection"],
        "test_scored_at_lock": False,
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return lock


def verify_analysis_lock(config):
    """Check actual data, model, protocol and code identities against the sealed lock."""
    root = _root(config)
    lock = json.loads((root / "analysis_lock.json").read_text())
    if lock["config_sha256"] != _config_hash(config):
        message = "Current configuration differs from the sealed test analysis"
        raise ValueError(message)
    required = {
        "plans/neutrino_study_protocol.txt": lock["protocol_sha256"],
        str(root / "cohort_manifest.parquet"): lock["cohort_manifest_sha256"],
        str(root / "baseline_selection.json"): lock["baseline_selection_sha256"],
        str(root / "training_reference.npz"): lock["training_reference_sha256"],
        str(root / "training_reference.json"): lock["training_reference_metadata_sha256"],
        str(root / "development_summary.json"): lock["development_summary_sha256"],
        **lock["code_sha256"],
    }
    required.update({item["path"]: item["sha256"] for item in lock["checkpoints"].values()})
    required.update({item["path"]: item["sha256"] for item in lock["observations"].values()})
    for path, expected in required.items():
        if shard_sha256(path) != expected:
            message = f"Sealed analysis input changed: {path}"
            raise ValueError(message)
    return lock


def _atomic_scores(path, records, values, **extra):
    """Publish complete scalar evidence for interruption-safe fixed-analysis continuation."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            event_ids=records.event_ids,
            configuration_ids=records.configuration_ids,
            retained_totals=records.retained_totals,
            target_totals=records.target_totals,
            **values,
            **extra,
        )
    temporary.replace(path)


def _test_resource_gate(config, summaries, development):
    """Account for development plus test cost against the per-seed resource cap."""
    resource = development["resource_gate"]
    for current, prior in zip(summaries, development["selected_validation"], strict=True):
        resource &= (
            current["elapsed_seconds"] + prior["total_evaluation_seconds"]
            <= config["training"]["max_evaluation_seconds_per_seed"]
        )
        resource &= (
            current["peak_process_tree_rss_gib"] <= config["training"]["max_process_rss_gib"]
        )
        resource &= (
            current["peak_allocated_gpu_gib"] is None
            or current["peak_allocated_gpu_gib"] <= config["training"]["max_gpu_allocated_gib"]
        )
    return bool(resource)


def score_locked_test(config, device_name):
    """Evaluate the first sealed test analysis; resume only that same unfinished analysis."""
    root = _root(config)
    if (root / "test_results.json").exists():
        message = "The independent test has already been evaluated; preserve its first result"
        raise FileExistsError(message)
    lock = verify_analysis_lock(config)
    lock_hash = shard_sha256(root / "analysis_lock.json")
    opened = root / "test_opened.json"
    if opened.exists():
        prior = json.loads(opened.read_text())
        if prior["analysis_lock_sha256"] != lock_hash:
            message = "An opened test cannot be reassigned to another analysis"
            raise ValueError(message)
        if psutil.pid_exists(prior["pid"]) and prior["pid"] != os.getpid():
            process = psutil.Process(prior["pid"])
            if abs(process.create_time() - prior["process_created"]) < 0.01:
                message = "The same locked test analysis is already running"
                raise RuntimeError(message)
    else:
        with opened.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "analysis_lock_sha256": lock_hash,
                    "pid": os.getpid(),
                    "process_created": psutil.Process().create_time(),
                    "created_unix_seconds": time.time(),
                    "interpretation": "one_locked_analysis; same_hash_incomplete_work_may_resume",
                },
                stream,
                indent=2,
            )
    output = root / "test_evidence"
    output.mkdir(exist_ok=True)
    device = _device(config, device_name)
    training, reference = _training_reference(config)
    testing = _dataset(config, "test")
    bank = _bank_from_selection(config, training, reference, device)
    donors = BoundedPhotonDonors(
        training,
        reference["retained_totals"],
        seed=config["evaluation"]["donor_seed"],
        cache_events=config["evaluation"]["donor_cache_events"],
    )
    summaries, paths = [], []
    for seed in config["training"]["seeds"]:
        path = output / f"seed_{seed}.npz"
        paths.append(path)
        if path.exists():
            summary_path = output / f"seed_{seed}.json"
            if not summary_path.exists():
                message = "Saved test evidence needs same-lock metadata recovery"
                raise RuntimeError(message)
            summaries.append(json.loads(summary_path.read_text()))
            continue
        begin = time.perf_counter()
        model = DensePhotonReference(config["model"]["channels"]).to(device)
        payload = torch.load(
            lock["checkpoints"][str(seed)]["path"], map_location="cpu", weights_only=True
        )
        model.load_state_dict(payload["state_dict"])
        records, values = stream_scores(
            testing,
            batch_size=config["training"]["batch_size"],
            device=device,
            model=model,
            baselines=bank,
        )
        donor_values, exact, donor_ids = stream_donor_scores(
            model,
            testing,
            donors,
            batch_size=config["training"]["batch_size"],
            device=device,
            draws=config["evaluation"]["donor_draws"],
        )
        if not exact[records.strata["retained_1_to_10"]].all():
            message = "Primary test donor matching is incomplete"
            raise RuntimeError(message)
        values["matched_donor"] = donor_values
        _atomic_scores(path, records, values, exact_donor=exact, donor_event_ids=donor_ids)
        result = _score_bundle(records, values, config, bank.winner, exact)
        result.update(
            {
                "seed": seed,
                "analysis_lock_sha256": lock_hash,
                "elapsed_seconds": time.perf_counter() - begin,
                "peak_process_tree_rss_gib": _rss_gib(),
                "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda"
                else None,
            }
        )
        write_json(output / f"seed_{seed}.json", result)
        summaries.append(result)
        LOGGER.info(
            "Test seed %s complete; low-count NLL %.6f",
            seed,
            result["scores"]["model"]["retained_1_to_10"]["nats_per_photon"],
        )
    records, _values = _read_scalar_scores(paths[0])
    aggregate = _aggregate_evidence(config, records, paths, bank.winner)
    primary = aggregate["contrasts"]["retained_1_to_10"]
    baseline, donor = primary[bank.winner], primary["matched_donor"]
    statistical = (
        baseline["descriptive_95_percent_interval"][1] < 0
        and donor["descriptive_95_percent_interval"][1] < 0
    )
    practical = (
        baseline["model_minus_control_nats_per_photon"]
        <= -config["evaluation"]["practical_improvement_nats_per_photon"]
    )
    development = json.loads((root / "development_summary.json").read_text())
    resource = _test_resource_gate(config, summaries, development)
    result = {
        "analysis_lock_sha256": lock_hash,
        "baseline": bank.winner,
        "scope": "independent_group_heldout_curated_simulation; physical_Poisson_not_certified",
        "aggregate": aggregate,
        "statistical_primary_gate": statistical,
        "practical_primary_gate": practical,
        "primary_quality_gate_passed": statistical and practical,
        "resource_gate_passed": bool(resource),
        "restricted_predictive_claim_supported": bool(statistical and practical and resource),
        "per_seed_test": summaries,
        "test_events": len(records.event_ids),
        "source_configuration_groups": len(np.unique(records.configuration_ids)),
        "model_seed_count": len(paths),
        "test_event_count_not_multiplied_by_seeds": True,
        "interpretation": "restricted_low_count_claim_requires_quality_and_resource_gates",
        "secondary_views_pending": config["test"]["secondary_split_probabilities"],
    }
    write_json(root / "test_results.json", result)
    testing.close()
    training.close()
    return result


def main():
    """Execute an explicit stage with a versioned, prospective configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("profile", "baselines", "train", "lock", "test"))
    parser.add_argument("--config", type=Path, default=Path("config/neutrino_study.yml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile-workers", type=int, choices=(0, 2))
    parser.add_argument("--profile-tag", default="engineering_profile")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.profile_workers is not None:
        if args.action != "profile":
            message = (
                "Worker overrides are engineering-only; revise the prospective config for training"
            )
            raise ValueError(message)
        config["training"]["num_workers"] = args.profile_workers
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    actions = {
        "profile": engineering_profile,
        "baselines": fit_study_baselines,
        "train": train_study,
        "test": score_locked_test,
    }
    job = {
        "pid": os.getpid(),
        "process_created": psutil.Process().create_time(),
        "action": args.action,
        "status": "running",
        "config_sha256": _config_hash(config),
    }
    write_json(_root(config) / "active_job.json", job)
    if args.action == "profile":
        engineering_profile(config, args.device, tag=args.profile_tag)
    elif args.action == "lock":
        lock_analysis(config)
    else:
        actions[args.action](config, args.device)
    job["status"] = "completed"
    write_json(_root(config) / "active_job.json", job)


if __name__ == "__main__":
    main()
