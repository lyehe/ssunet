"""Continuous two-detector inference with real temporal halos for motion inspection."""

from __future__ import annotations

import logging

import numpy as np
import torch

from .models.confocal_multichannel import channel_log_probabilities

LOGGER = logging.getLogger(__name__)


def split_motion_pool(pool, *, seed, first_time):
    """Split the same acquisition pool once, before any overlapping inference windows."""
    if pool.ndim != 5 or pool.dtype != np.uint8:
        message = "Expected CTZYX uint8 photon counts"
        raise ValueError(message)
    retained = np.zeros_like(pool)
    for channel in range(pool.shape[0]):
        for time, frame in enumerate(pool[channel]):
            present = frame != 0
            rng = np.random.default_rng([seed, channel, first_time + time])
            retained[channel, time][present] = rng.binomial(frame[present], 0.5).astype(np.uint8)
    return retained, pool - retained


@torch.inference_mode()
def _logits(model, counts, device):
    values = torch.from_numpy(np.array(counts, copy=True, order="C"))[None].to(device)
    return model(values)


@torch.inference_mode()
def temporal_logits(model, counts, *, frames, halo, core, device):
    """Stitch valid-center logits, without independently normalizing adjacent chunks."""
    required = 2 * (len(model.network.encoder) + len(model.network.decoder))
    if (
        halo < required
        or counts.shape[1] != frames + 2 * halo
        or counts.shape[0] != model.channels
        or frames < 1
        or core < 1
    ):
        message = "Provide both detectors, the complete real temporal halo, and positive core sizes"
        raise ValueError(message)
    output = np.empty((counts.shape[0], frames, *counts.shape[2:]), dtype=np.float32)
    for start in range(0, frames, core):
        length = min(core, frames - start)
        logits = _logits(model, counts[:, start : start + length + 2 * halo], device)
        output[:, start : start + length] = logits[0, :, halo : halo + length].cpu().numpy()
        LOGGER.info("Temporal inference %d/%d frames", start + length, frames)
    return output


@torch.inference_mode()
def temporal_control(model, counts, *, halo, core, device):
    """Check chunk agreement and distinguish static-input padding artifacts from motion."""
    first = _logits(model, counts[:, : core + 2 * halo], device)
    offset = core // 2
    second = _logits(model, counts[:, offset : offset + core + 2 * halo], device)
    overlap = (
        (first[:, :, halo + offset : halo + core] - second[:, :, halo : halo + core - offset])
        .abs()
        .max()
        .item()
    )
    frame = counts[:, counts.shape[1] // 2]
    static_long = np.repeat(frame[:, None], core + 2 * halo, axis=1)
    long_logits = _logits(model, static_long, device)[:, :, halo : halo + core]
    static_error = (long_logits - long_logits[:, :, core // 2 : core // 2 + 1]).abs().max().item()
    static_short = np.repeat(frame[:, None], core, axis=1)
    short_logits = _logits(model, static_short, device)
    short_centered = short_logits - short_logits.mean(dim=(-3, -2, -1), keepdim=True)
    padding_error = (
        (short_centered - short_centered[:, :, core // 2 : core // 2 + 1]).abs().max().item()
    )
    if max(overlap, static_error) > 2e-4:
        message = "Temporal halos failed the overlap/static-input consistency check"
        raise ValueError(message)
    return {
        "status": "passed",
        "overlapping_core_max_logit_difference": overlap,
        "static_input_halo_core_max_logit_difference": static_error,
        "static_input_without_halo_max_shape_logit_difference": padding_error,
        "scope": "Software consistency controls, not biological ground truth",
    }


@torch.inference_mode()
def normalized_motion_probabilities(logits, device):
    """Normalize each detector once over the complete continuous core clip."""
    values = torch.from_numpy(logits)[None].to(device)
    q = channel_log_probabilities(values).exp()[0].cpu().numpy()
    mass = q.sum(axis=(1, 2, 3, 4), dtype=np.float64)
    q /= mass[:, None, None, None, None]
    return q
