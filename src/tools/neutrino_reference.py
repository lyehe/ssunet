"""Build and inspect controlled photon shards without simulation-truth supervision.

Use ``python -m src.tools.neutrino_reference --help`` from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

if __package__ == "src.tools":
    from ..ssunet.datasets.neutrino import (
        NeutrinoPhotonDataset,
        partition_configurations,
        shard_sha256,
        write_photon_shard,
    )
    from ..ssunet.models.neutrino_dense import DensePhotonReference
    from ..ssunet.photon_reference import (
        audit_poisson_thinning,
        photon_count_log_probabilities,
        photon_count_nll,
    )
else:
    from ssunet.datasets.neutrino import (
        NeutrinoPhotonDataset,
        partition_configurations,
        shard_sha256,
        write_photon_shard,
    )
    from ssunet.models.neutrino_dense import DensePhotonReference
    from ssunet.photon_reference import (
        audit_poisson_thinning,
        photon_count_log_probabilities,
        photon_count_nll,
    )


def _write_json(path: Path, content: dict) -> None:
    """Write a machine-readable companion artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_analytic_reference(path: Path, config: dict) -> dict:
    """Generate an exact Poisson toy reference with geometry-dependent light."""
    generator = np.random.default_rng(config["seed"])
    settings = config["analytic"]
    strings = settings["strings"]
    per_string = settings["sensors_per_string"]
    phi = np.arange(strings) * 2 * np.pi / strings
    positions = np.column_stack(
        [
            np.repeat(120 * np.cos(phi), per_string),
            np.repeat(120 * np.sin(phi), per_string),
            np.tile(np.linspace(-1000, -500, per_string), strings),
        ]
    )
    edges = np.linspace(*config["time_window_ns"], config["time_bins"] + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    levels = np.asarray(settings["expected_signal_photons"], dtype=float)
    background = float(settings["expected_background_photons"])
    if np.any(levels < 0) or background < 0 or not np.isfinite(levels).all():
        message = "Expected photon levels must be finite and nonnegative"
        raise ValueError(message)
    event_ids = np.arange(config["events"], dtype=np.int64)
    configuration_ids = event_ids // len(levels)
    n_groups = int(configuration_ids[-1]) + 1
    sources = generator.uniform([-150, -150, -1050], [150, 150, -450], size=(n_groups, 3))
    photon_sensors = []
    photon_times = []
    offsets = [0]
    expected_totals = []
    for event, group in zip(event_ids, configuration_ids, strict=True):
        distance = np.linalg.norm(positions - sources[group], axis=1)
        peak = 500 + distance / 0.225
        delay = centers[None, :] - peak[:, None]
        temporal = np.exp(-0.5 * (delay / 60) ** 2)
        temporal += 0.35 * np.exp(-np.maximum(delay, 0) / 200) * (delay >= 0)
        spatial = np.exp(-distance / 300) / (distance**2 + 50**2)
        pdf = spatial[:, None] * temporal
        pdf /= pdf.sum()
        intensity = levels[event % len(levels)] * pdf + background / pdf.size
        total = float(intensity.sum())
        expected_totals.append(total)
        # Poisson total + iid location draws is exactly a multivariate Poisson.
        number = generator.poisson(total)
        locations = (
            generator.choice(pdf.size, size=number, p=intensity.ravel() / total)
            if total > 0
            else np.empty(0, dtype=np.int64)
        )
        sensor, time_bin = np.divmod(locations, config["time_bins"])
        times = edges[time_bin] + generator.random(number) * np.diff(edges)[time_bin]
        photon_sensors.append(sensor.astype(np.int32))
        photon_times.append(times)
        offsets.append(offsets[-1] + number)
    write_photon_shard(
        path,
        offsets=np.asarray(offsets, dtype=np.int64),
        sensor_index=np.concatenate(photon_sensors),
        time_ns=np.concatenate(photon_times),
        positions_m=positions,
        live=np.ones(len(positions), dtype=bool),
        event_ids=event_ids,
        configuration_ids=configuration_ids,
        partition=partition_configurations(configuration_ids, config["seed"]),
        time_edges_ns=edges,
        provenance={
            "kind": "analytic_poisson_toy_not_neutrino_transport",
            "seed": config["seed"],
            "config": config,
            "unit_photon_detections": True,
            "clock": "fixed acquisition window before photon sampling",
            "conditional_poisson_assumption": "exact by construction",
        },
    )
    np.savez_compressed(
        path.with_suffix(".evaluation_only.npz"),
        event_id=event_ids,
        configuration_id=configuration_ids,
        latent_source_position_m=sources,
        expected_total_photons=np.asarray(expected_totals),
    )
    return inspect_shard(path)


def _match_photon_sensor(string, sensor, coordinates, positions, geometry_strings, lookup):
    """Resolve local/global ID differences using fixed geometry coordinates."""
    if not float(string).is_integer() or not float(sensor).is_integer():
        message = "Source sensor identities are not integers"
        raise ValueError(message)
    key = (string, sensor)
    if key not in lookup:
        matching = np.flatnonzero(
            np.isclose(positions, coordinates, rtol=0, atol=1e-5).all(axis=1)
            & (geometry_strings == string)
        )
        if len(matching) != 1:
            message = "Photon coordinates do not uniquely identify a sensor in the fixed geometry"
            raise ValueError(message)
        lookup[key] = int(matching[0])
    index = lookup[key]
    if not np.allclose(positions[index], coordinates, rtol=0, atol=1e-5):
        message = "One source sensor identity maps to inconsistent positions"
        raise ValueError(message)
    return index


def import_prometheus_sqlite(
    database: Path, geometry_path: Path, output: Path, config: dict
) -> dict:
    """Import unit photon rows and all sampled truth-table events, including empty ones."""
    import pandas as pd

    geometry = pd.read_parquet(geometry_path)
    positions = geometry[["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]].to_numpy()
    string_column = "sensor_string_id" if "sensor_string_id" in geometry else "string_id"
    keys = list(zip(geometry[string_column], geometry["sensor_id"], strict=True))
    if len(set(keys)) != len(keys):
        message = "Geometry has duplicate string/sensor identities"
        raise ValueError(message)
    # The published geometry uses global sensor IDs, while photon rows use
    # IDs local to a string. Match fixed positions instead of assuming equality.
    lookup = {}
    geometry_strings = geometry[string_column].to_numpy()
    sensor_arrays, time_arrays, offsets, truth = [], [], [0], []
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(photons)")}
        required = {
            "event_no",
            "string_id",
            "sensor_id",
            "t",
            "sensor_pos_x",
            "sensor_pos_y",
            "sensor_pos_z",
        }
        if columns != required:
            message = "Unverified photon schema or additional weights/response fields"
            raise ValueError(message)
        all_ids = np.asarray(
            connection.execute("SELECT event_no FROM mc_truth ORDER BY event_no").fetchall(),
            dtype=np.int64,
        ).ravel()
        if config["events"] > len(all_ids):
            message = "Requested more distinct events than the source contains"
            raise ValueError(message)
        event_ids = np.sort(
            np.random.default_rng(config["seed"]).choice(all_ids, config["events"], replace=False)
        )
        truth_columns = [row[1] for row in connection.execute("PRAGMA table_info(mc_truth)")]
        for event_id in event_ids:
            rows = connection.execute(
                "SELECT string_id,sensor_id,t,sensor_pos_x,sensor_pos_y,sensor_pos_z "
                "FROM photons WHERE event_no=? ORDER BY t,string_id,sensor_id",
                (int(event_id),),
            ).fetchall()
            source_indices, times = [], []
            for string, sensor, time_ns, x, y, z in rows:
                index = _match_photon_sensor(
                    string, sensor, [x, y, z], positions, geometry_strings, lookup
                )
                source_indices.append(index)
                times.append(time_ns)
            sensor_arrays.append(np.asarray(source_indices, dtype=np.int32))
            time_arrays.append(np.asarray(times, dtype=np.float64))
            offsets.append(offsets[-1] + len(rows))
            record = connection.execute(
                "SELECT * FROM mc_truth WHERE event_no=?", (int(event_id),)
            ).fetchone()
            truth.append(dict(zip(truth_columns, record, strict=True)))
    edges = np.linspace(*config["time_window_ns"], config["time_bins"] + 1)
    provenance = {
        "kind": "public_prometheus_pone_water_photon_simulation",
        "source_url": "https://sid.erda.dk/share_redirect/GIt0hlG9qI",
        "source_definition": "https://github.com/graphnet-team/graphnet/blob/main/src/graphnet/datasets/prometheus_datasets.py",
        "source_database": database.name,
        "source_database_bytes": database.stat().st_size,
        "source_events": len(all_ids),
        "geometry_sha256": shard_sha256(geometry_path),
        "geometry_sensor_keys": [[int(string), int(sensor)] for string, sensor in keys],
        "observed_source_sensor_mapping": [
            [int(string), int(sensor), index] for (string, sensor), index in sorted(lookup.items())
        ],
        "sensor_mapping": "fixed positions and string IDs; source local IDs preserved in mapping",
        "unit_photon_detections": True,
        "detector_medium": "water",
        "detector_response": "source contains photon rows; no charge or waveform fields",
        "selection": "public curated archive merged_cut.db; full upstream selection not certified",
        "conditional_poisson_assumption": "not certified from a single physical MC realization",
        "clock": "original source t retained; no alignment from A+B",
        "seed": config["seed"],
        "pilot_event_selection_uses_photon_counts": False,
        "upstream_selection_uses_photon_counts": "unknown",
    }
    write_photon_shard(
        output,
        offsets=np.asarray(offsets, dtype=np.int64),
        sensor_index=np.concatenate(sensor_arrays),
        time_ns=np.concatenate(time_arrays),
        positions_m=positions,
        live=np.ones(len(positions), dtype=bool),
        event_ids=event_ids,
        configuration_ids=event_ids.copy(),
        partition=partition_configurations(event_ids, config["seed"]),
        time_edges_ns=edges,
        provenance=provenance,
    )
    _write_json(output.with_suffix(".evaluation_only.json"), {"events": truth})
    return inspect_shard(output)


def inspect_shard(path: Path) -> dict:
    """Measure sparsity, partition sizes, conservation and evaluation reproducibility."""
    dataset = NeutrinoPhotonDataset(path, split="all", training=False)
    counts, outside, frozen_inputs = [], [], []
    empty_input_splits = 0
    empty_target_splits = 0
    conserved = True
    shared_bins = False
    maximum_bin_count = 0
    for index in range(len(dataset)):
        histogram, dropped = dataset.counts(index)
        item = dataset[index]
        conserved &= bool(np.array_equal((item["input"] + item["target"]).numpy(), histogram))
        shared_bins |= bool(((item["input"] > 0) & (item["target"] > 0)).any())
        maximum_bin_count = max(maximum_bin_count, int(histogram.max()))
        counts.append(int(histogram.sum()))
        outside.append(dropped)
        empty_input_splits += int(item["input"].sum()) == 0
        empty_target_splits += int(item["target"].sum()) == 0
        if index < 20:
            frozen_inputs.append(item["input"].clone())
    n = np.asarray(counts)
    with h5py.File(path, "r") as handle:
        partition = handle["events/partition"][:]
    frozen = all(
        np.array_equal(item.numpy(), dataset[index]["input"].numpy())
        for index, item in enumerate(frozen_inputs)
    )
    report = {
        "path": str(path),
        "sha256": shard_sha256(path),
        "provenance": dataset.provenance,
        "events": len(dataset),
        "sensors": len(dataset.live),
        "time_bins": len(dataset.edges_ns) - 1,
        "partitions": {
            name: int(np.sum(partition == value))
            for value, name in enumerate(["train", "validation", "test"])
        },
        "photons_in_window": int(n.sum()),
        "photons_outside_window": int(np.sum(outside)),
        "empty_observations": int(np.sum(n == 0)),
        "observations_with_1_to_10_photons": int(np.sum((n > 0) & (n <= 10))),
        "photons_per_event_quantiles": np.quantile(n, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        "empty_input_splits": empty_input_splits,
        "empty_target_splits": empty_target_splits,
        "maximum_count_in_one_bin": maximum_bin_count,
        "input_and_target_may_share_a_bin": shared_bins,
        "split_conservation_passed": conserved,
        "evaluation_draws_reproducible": frozen,
        "ground_truth_used_by_dataset": False,
    }
    _write_json(path.with_suffix(".inspection.json"), report)
    return report


def _validation_scores(model, training, validation, device, batch_size) -> dict:
    """Score frozen validation photons; estimate the prior on training events only."""
    support = np.broadcast_to(
        training.live[:, None], (len(training.live), len(training.edges_ns) - 1)
    )
    prior = support.astype(np.float64) * 0.5
    for index in range(len(training)):
        prior += training.counts(index)[0]
    prior = torch.as_tensor(prior, dtype=torch.float32, device=device)
    prior_log = (prior / prior.sum()).clamp_min(torch.finfo(torch.float32).tiny).log()
    nll = {"model": 0.0, "uniform": 0.0, "training_prior": 0.0}
    photons = 0
    events_with_target = 0
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(validation, batch_size=batch_size):
            counts = batch["input"].to(device)
            target = batch["target"].to(device)
            live = batch["live"].to(device)
            logits = model(counts, batch["positions_m"].to(device), live, batch["p"].to(device))
            log_probability = photon_count_log_probabilities(logits, live)
            safe_log_probability = log_probability.masked_fill(target == 0, 0)
            batch_photons = int(target.sum())
            photons += batch_photons
            events_with_target += int((target.sum(dim=(1, 2)) > 0).sum())
            nll["model"] += float(-(target * safe_log_probability).sum())
            nll["uniform"] += batch_photons * math.log(int(support.sum()))
            nll["training_prior"] += float(-(target * prior_log).sum())
    return {
        "partition": "validation_only_test_not_scored",
        "held_out_photons": photons,
        "events_with_target": events_with_target,
        "nats_per_photon": {
            name: value / photons if photons else None for name, value in nll.items()
        },
        "prior_pseudocount_per_valid_bin": 0.5,
    }


def _reference_device(device_name: str) -> torch.device:
    """Resolve an explicitly selected device without silently changing backends."""
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            message = "CUDA was requested but is unavailable; choose --device cpu explicitly"
            raise RuntimeError(message)
        torch.cuda.set_device(device)
    return device


def profile_reference(
    path: Path,
    config: dict,
    report_path: Path,
    device_name: str = "cuda:0",
    steps: int | None = None,
) -> dict:
    """Run a bounded training smoke test and measure complete batch throughput."""
    settings = config["model"]
    steps = settings["profile_steps"] if steps is None else steps
    if steps <= 0:
        message = "Profile must contain at least one optimizer step"
        raise ValueError(message)
    device = _reference_device(device_name)
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    training = NeutrinoPhotonDataset(
        path, "train", p=config["split_probability"], seed=config["seed"]
    )
    validation = NeutrinoPhotonDataset(
        path, "validation", p=config["split_probability"], seed=config["seed"]
    )
    loader = DataLoader(
        training,
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    model = DensePhotonReference(channels=settings["channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    iterator = iter(loader)
    epoch = 0
    warmup = min(10, steps)
    n_events = 0
    losses = []
    begin = None
    for step in range(steps + warmup):
        if step == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            begin = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            training.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
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
            message = "Non-finite reference training loss"
            raise RuntimeError(message)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step >= warmup:
            n_events += len(counts)
            losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - begin
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    result = {
        "scope": "bounded_optimizer_and_throughput_smoke_test_not_full_three_seed_experiment",
        "dataset": str(path),
        "dataset_sha256": shard_sha256(path),
        "source_kind": training.provenance["kind"],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "bfloat16_with_fp32_normalization" if device.type == "cuda" else "float32",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "profiled_optimizer_steps": steps,
        "warmup_optimizer_steps": warmup,
        "profiled_events": n_events,
        "profiled_elapsed_s": elapsed,
        "end_to_end_training_events_per_second": n_events / elapsed,
        "peak_allocated_gpu_gib": peak / 2**30 if peak is not None else None,
        "peak_reserved_gpu_gib": reserved / 2**30 if reserved is not None else None,
        "memory_target_gib": settings["memory_target_gib"],
        "memory_target_met": peak < settings["memory_target_gib"] * 2**30
        if peak is not None
        else None,
        "all_losses_and_gradients_finite": True,
        "first_profiled_batch_raw_nll_per_event": losses[0],
        "last_profiled_batch_raw_nll_per_event": losses[-1],
        "validation": _validation_scores(
            model, training, validation, device, settings["batch_size"]
        ),
        "test_partition_used_for_model_selection": False,
    }
    _write_json(report_path, result)
    return result


def main() -> None:
    """Run one explicit acquisition, generation or validation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/neutrino_reference.yml"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("analytic", "import-prometheus", "inspect", "audit", "profile"):
        command = commands.add_parser(name)
        command.add_argument("path", type=Path)
        if name == "import-prometheus":
            command.add_argument("--database", type=Path, required=True)
            command.add_argument("--geometry", type=Path, required=True)
        if name == "profile":
            command.add_argument("--report", type=Path, required=True)
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--steps", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.command == "analytic":
        result = create_analytic_reference(args.path, config)
    elif args.command == "import-prometheus":
        result = import_prometheus_sqlite(args.database, args.geometry, args.path, config)
    elif args.command == "inspect":
        result = inspect_shard(args.path)
    elif args.command == "profile":
        result = profile_reference(args.path, config, args.report, args.device, args.steps)
    else:
        settings = config["statistics"]
        result = audit_poisson_thinning(
            seed=config["seed"],
            samples=settings["realizations_per_rate"],
            sigma_limit=settings["sigma_limit"],
        )
        _write_json(args.path, result)
        if not result["passed"]:
            message = "Poisson thinning checks failed; inspect the saved report"
            raise RuntimeError(message)
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
