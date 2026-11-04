"""Photon-pool isolation and detector-wise scoring for joint-channel confocal trials."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .confocal_inference import sha256_file, write_json
from .models.confocal_multichannel import channel_log_probabilities, split_all_channels

LOGGER = logging.getLogger(__name__)


def thin_frame(frame, probability, *, seed, channel, time):
    """Thin nonzero counts without allocating a dense int64 random draw for the frame."""
    if frame.dtype != np.uint8 or not 0 < probability < 1:
        message = "Expected uint8 photon counts and a probability strictly between zero and one"
        raise ValueError(message)
    retained = np.zeros_like(frame)
    present = frame != 0
    generator = np.random.default_rng(np.random.SeedSequence([seed, channel, time]))
    retained[present] = generator.binomial(frame[present], probability).astype(np.uint8)
    return retained


def prepare_training_pool(counts, destination, settings, raw_sha256):
    """Permanently exclude evaluation photons from every optimization input and target."""
    destination = Path(destination)
    identity = {"settings": settings, "raw_sha256": raw_sha256, "shape": list(counts.shape)}
    manifest_path = destination.with_suffix(".json")
    if destination.exists() or manifest_path.exists():
        if not destination.exists() or not manifest_path.exists():
            message = "Incomplete photon pool; choose a fresh output directory"
            raise FileExistsError(message)
        manifest = json.loads(manifest_path.read_text())
        values = np.load(destination, mmap_mode="r", allow_pickle=False)
        if (
            manifest["identity"] != identity
            or list(values.shape) != identity["shape"]
            or values.dtype != np.uint8
            or sha256_file(destination) != manifest["sha256"]
        ):
            message = "Training photon pool identity or content changed"
            raise ValueError(message)
        return values, manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial.npy")
    pool = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=counts.shape)
    totals = np.zeros((counts.shape[0], counts.shape[1], 2), dtype=np.int64)
    for channel in range(counts.shape[0]):
        for time in range(counts.shape[1]):
            raw = np.asarray(counts[channel, time])
            kept = thin_frame(
                raw,
                settings["training_probability"],
                seed=settings["seed"],
                channel=channel,
                time=time,
            )
            pool[channel, time] = kept
            totals[channel, time] = [raw.sum(dtype=np.int64), kept.sum(dtype=np.int64)]
            if (time + 1) % 128 == 0:
                LOGGER.info(
                    "Training photon pool: detector %d, %d/%d times",
                    channel,
                    time + 1,
                    counts.shape[1],
                )
    pool.flush()
    del pool
    partial.replace(destination)
    manifest = {
        "identity": identity,
        "sha256": sha256_file(destination),
        "raw_and_training_photons_by_channel_time": totals.tolist(),
        "scope": "Only this pool enters training; raw minus pool is evaluation-only",
    }
    write_json(manifest_path, manifest)
    return np.load(destination, mmap_mode="r", allow_pickle=False), manifest


def validation_cases(raw, pool, config):
    """Freeze phase-stratified inputs while keeping evaluation targets outside the pool."""
    patch = config["training"]["patch_tzyx"]
    cases = []
    for phase, time in config["evaluation"]["phases"].items():
        for position in config["evaluation"]["spatial_starts"]:
            start = [time, *position]
            if any(
                a < 0 or a + n > have
                for a, n, have in zip(start, patch, raw.shape[1:], strict=True)
            ):
                message = "A fixed evaluation case lies outside the source crop"
                raise ValueError(message)
            selection = (slice(None), *(slice(a, a + n) for a, n in zip(start, patch, strict=True)))
            observed = np.asarray(raw[selection])
            train = np.asarray(pool[selection])
            if np.any(train > observed):
                message = "The training photon pool must be a subset of observed counts"
                raise ValueError(message)
            rng = np.random.default_rng([config["evaluation"]["split_seed"], len(cases)])
            a = rng.binomial(train, config["training"]["split_probability"]).astype(np.uint8)
            cases.append(
                {
                    "index": len(cases),
                    "phase": phase,
                    "start_tzyx": start,
                    "input": torch.from_numpy(a[None]),
                    "target": torch.from_numpy((observed - train)[None]),
                    "observed": np.array(observed, copy=True),
                }
            )
    return cases


def training_batch(pool, config, step, microstep):
    """Address each patch independently of microbatch grouping and Torch's RNG."""
    options = config["training"]
    first = ((step - 1) * options["accumulation_steps"] + microstep) * options["batch_size"]
    pairs = [
        split_all_channels(
            pool,
            options["patch_tzyx"],
            seed=options["photon_split_seed"],
            index=first + offset,
            batch_size=1,
            p=options["split_probability"],
        )
        for offset in range(options["batch_size"])
    ]
    return tuple(torch.cat([pair[side] for pair in pairs]) for side in (0, 1))


def score_log_probabilities(logq, target, case):
    """Accumulate photon NLL in float64, retaining empty-detector cases explicitly."""
    if logq.shape != target.shape or logq.ndim != 6 or logq.shape[0] != 1:
        message = "Expected matching [1,C,T,Z,Y,X] log probabilities and held-out counts"
        raise ValueError(message)
    flat, target = logq.flatten(2), target.to(logq.device).flatten(2)
    total = target.sum(-1)[0].tolist()
    # The reduction stays on the GPU; only per-detector scalar evidence is copied.
    nll = (-(target.double() * flat.double()).sum(-1))[0].tolist()
    mass = flat.exp().double().sum(-1)[0].tolist()
    return {
        "index": case["index"],
        "phase": case["phase"],
        "start_tzyx": case["start_tzyx"],
        "channels": [
            {
                "channel": c,
                "target_photons": int(n),
                "total_nll": value,
                "photon_nll": value / n if n else None,
                "probability_mass": m,
            }
            for c, (n, value, m) in enumerate(zip(total, nll, mass, strict=True))
        ],
    }


def _pooled_score(rows):
    photons = sum(row["target_photons"] for row in rows)
    nll = sum(row["total_nll"] for row in rows)
    return {
        "target_photons": photons,
        "total_nll": nll,
        "photon_nll": nll / photons if photons else None,
        "empty_cases": sum(row["target_photons"] == 0 for row in rows),
    }


def _phase_score(rows, phase):
    selected = [c for row in rows if row["phase"] == phase for c in row["channels"]]
    return {
        **_pooled_score(selected),
        "per_channel": [
            _pooled_score([c for c in selected if c["channel"] == channel])
            for channel in range(len(rows[0]["channels"]))
        ],
    }


def summarize_scores(rows):
    """Separate channel and time-phase scores so bright early frames cannot hide late failures."""
    flat = [c for row in rows for c in row["channels"]]
    channels = [
        _pooled_score([c for c in flat if c["channel"] == channel])
        for channel in range(len(rows[0]["channels"]))
    ]
    valid = [c["photon_nll"] for c in channels if c["photon_nll"] is not None]
    return {
        **_pooled_score(flat),
        "per_channel": channels,
        "channel_mean_nll": float(np.mean(valid)) if valid else None,
        "per_phase": {
            phase: _phase_score(rows, phase)
            for phase in dict.fromkeys(row["phase"] for row in rows)
        },
        "max_probability_mass_error": max(abs(c["probability_mass"] - 1) for c in flat),
        "cases": rows,
    }


@torch.inference_mode()
def evaluate_model(model, cases, device):
    """Score only reserved photons; no evaluation target enters the network."""
    model.eval()
    rows = []
    for case in cases:
        logits = model(case["input"].to(device))
        logq = channel_log_probabilities(logits)
        rows.append(score_log_probabilities(logq, case["target"], case))
    result = summarize_scores(rows)
    if result["photon_nll"] is None or result["max_probability_mass_error"] > 1e-5:
        message = "Evaluation needs target photons and unit-mass detector distributions"
        raise ValueError(message)
    return result


def gaussian_probabilities(inputs, sigma, uniform_mixture):
    """Use a fixed input-only smoother with a small uniform floor for unobserved locations."""
    if not 0 < uniform_mixture < 1:
        message = "The uniform mixture must be strictly between zero and one"
        raise ValueError(message)
    q = np.empty(inputs.shape, dtype=np.float32)
    for channel, observed in enumerate(inputs):
        values = gaussian_filter(observed.astype(np.float32), sigma, mode="reflect", truncate=3)
        total = float(values.sum(dtype=np.float64))
        q[channel] = (
            (1 - uniform_mixture) * values / total + uniform_mixture / values.size
            if total
            else np.full_like(values, 1 / values.size)
        )
    return q


def evaluate_baselines(cases, config):
    """Score predeclared uniform, spatial-Gaussian and spatiotemporal-Gaussian controls."""
    options = config["evaluation"]
    spatial = options["gaussian_spatial_sigma_zyx"]
    methods = {
        "uniform": None,
        "gaussian3d": [0, *spatial],
        "gaussian4d": [options["gaussian_temporal_sigma"], *spatial],
    }
    result = {}
    for name, sigma in methods.items():
        rows = []
        for case in cases:
            values = case["input"].numpy()[0]
            logq = (
                torch.full(case["input"].shape, -math.log(math.prod(values.shape[1:])))
                if sigma is None
                else torch.from_numpy(
                    gaussian_probabilities(values, sigma, options["uniform_mixture"])
                )
                .log()
                .unsqueeze(0)
            )
            rows.append(score_log_probabilities(logq, case["target"], case))
        result[name] = summarize_scores(rows)
    return result
