"""Input-only motion controls and photon likelihoods, without clean image targets."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_motion_probability(first, second, *, sigma, halo, frames, uniform_mixture=0.01):
    """Average PDFs from the same complementary photon inputs used by the network.

    Inputs have TZYX order for one detector, including real temporal context.
    Spatial reflection matches the earlier fixed Gaussian controls. Crop the
    filtered values before global normalization so halo photons are not scored.
    """
    if (
        first.ndim != 4
        or first.shape != second.shape
        or len(sigma) != 4
        or any(not math.isfinite(s) or s < 0 for s in sigma)
        or not 0 < uniform_mixture < 1
        or frames < 1
        or halo < int(3 * sigma[0] + 0.5)
        or first.shape[0] != frames + 2 * halo
    ):
        message = "Provide matching TZYX inputs, positive core length and enough real temporal halo"
        raise ValueError(message)
    shape = (frames, *first.shape[1:])
    result = np.zeros(shape, dtype=np.float64)
    for observed in (first, second):
        if observed.dtype.kind not in "ui" or (observed.dtype.kind == "i" and np.any(observed < 0)):
            message = "Gaussian inputs must be nonnegative integer photon counts"
            raise ValueError(message)
        filtered = gaussian_filter(observed.astype(np.float32), sigma, mode="reflect", truncate=3)
        core = filtered[halo : halo + frames]
        total = core.sum(dtype=np.float64)
        if total:
            result += 0.5 * ((1 - uniform_mixture) * core / total + uniform_mixture / core.size)
        else:
            result += 0.5 / core.size
    result /= result.sum(dtype=np.float64)
    return result.astype(np.float32)


def factorized_motion_probability(probability):
    """Remove space-time interactions while preserving temporal and spatial marginals.

    This diagnostic freezes spatial shape, retaining the model's own time-varying
    total rate. It does not establish whether any particular filament moves.
    """
    if probability.ndim != 4 or not np.isfinite(probability).all() or np.any(probability < 0):
        message = "Expected a finite nonnegative TZYX photon probability map"
        raise ValueError(message)
    total = probability.sum(dtype=np.float64)
    if total <= 0:
        message = "A probability map must have positive total mass"
        raise ValueError(message)
    temporal = probability.sum(axis=(1, 2, 3), dtype=np.float64) / total
    spatial = probability.sum(axis=0, dtype=np.float64) / total
    return (temporal[:, None, None, None] * spatial[None]).astype(np.float32)


def photon_log_score(probability, reserved, *, block=16):
    """Sum photon log losses in float64; reserved observations are not image GT."""
    if (
        probability.shape != reserved.shape
        or probability.ndim != 4
        or reserved.dtype.kind not in "ui"
        or np.any(reserved < 0)
        or block < 1
    ):
        message = "Provide matching TZYX probabilities and nonnegative integer reserved counts"
        raise ValueError(message)
    if not np.isfinite(probability).all() or np.any(probability < 0):
        message = "Photon probability maps must be finite and nonnegative"
        raise ValueError(message)
    total = probability.sum(dtype=np.float64)
    if total <= 0:
        message = "A probability map must have positive total mass"
        raise ValueError(message)
    blocks = []
    for start in range(0, probability.shape[0], block):
        target = reserved[start : start + block]
        present = target > 0
        locations = probability[start : start + block][present].astype(np.float64)
        if np.any(locations <= 0):
            message = "Every scored photon must have strictly positive predicted probability"
            raise ValueError(message)
        loss = -float(
            np.dot(target[present].astype(np.float64), np.log(locations) - math.log(total))
        )
        photons = int(target.sum(dtype=np.int64))
        blocks.append({"start": start, "photons": photons, "negative_log_likelihood": loss})
    loss = math.fsum(row["negative_log_likelihood"] for row in blocks)
    photons = sum(row["photons"] for row in blocks)
    return {
        "photons": photons,
        "negative_log_likelihood": loss,
        "nats_per_photon": loss / photons if photons else None,
        "probability_mass": float(total),
        "blocks": blocks,
    }
