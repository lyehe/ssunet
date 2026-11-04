"""Poisson-TV denoising with physical vertical neighbors and a primal-dual gap."""

import math

import numpy as np
import torch


def poisson_prox(values, observed, step):
    """Proximal map of step*sum(x - A*log(x)), including A=0 and x>=0."""
    shifted = values - step
    root = (shifted.square() + 4 * step * observed).sqrt()
    positive = 0.5 * (shifted + root)
    stable_negative = (2 * step * observed) / (root - shifted).clamp_min(
        torch.finfo(values.dtype).tiny
    )
    return torch.where(shifted >= 0, positive, stable_negative)


def tv_gradient(values, vertical_weights):
    """Compute time differences and weighted differences within actual detector strings."""
    temporal = values[..., 1:] - values[..., :-1]
    vertical = (values[:, 1:] - values[:, :-1]) * vertical_weights[None, :, None]
    return temporal, vertical


def tv_adjoint(temporal, vertical, vertical_weights, shape):
    """Apply the exact adjoint of the two discrete physical-gradient operators."""
    result = temporal.new_zeros(shape)
    result[..., :-1] -= temporal
    result[..., 1:] += temporal
    weighted = vertical * vertical_weights[None, :, None]
    result[:, :-1] -= weighted
    result[:, 1:] += weighted
    return result


def _physical_order(positions_m, live, vertical_strength):
    positions = positions_m.detach().cpu().double().numpy()
    order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
    ordered = positions[order]
    same_string = np.isclose(ordered[1:, :2], ordered[:-1, :2], atol=1e-4, rtol=0).all(-1)
    distance = np.abs(ordered[1:, 2] - ordered[:-1, 2])
    valid = same_string & (distance > 1e-4)
    spacing = float(np.median(distance[valid])) if valid.any() else 1.0
    weights = np.where(valid, spacing / np.maximum(distance, 1e-4), 0.0)
    indices = torch.as_tensor(order, dtype=torch.long, device=live.device)
    active = live[indices]
    result = torch.as_tensor(weights, device=live.device, dtype=torch.float64) * vertical_strength
    result *= active[1:] & active[:-1]
    return indices, torch.argsort(indices), result


def poisson_tv_gap(values, observed, temporal, vertical, weights, regularization):
    """Compute a certified gap using a feasible rescaling of the dual variables."""
    kt = tv_adjoint(temporal, vertical, weights, values.shape)
    dual_argument = -kt
    maximum = dual_argument.amax((1, 2), keepdim=True).clamp_min(1)
    scale = (1 - 1e-12) / maximum
    dual_argument = dual_argument * scale
    safe_values = values.clamp_min(torch.finfo(values.dtype).tiny)
    data = (values - observed * safe_values.log()).sum((1, 2))
    dt, dz = tv_gradient(values, weights)
    primal = data + regularization * (dt.abs().sum((1, 2)) + dz.abs().sum((1, 2)))
    safe_counts = observed.clamp_min(1)
    dual = (observed - observed * safe_counts.log() + observed * torch.log1p(-dual_argument)).sum(
        (1, 2)
    )
    return (primal - dual).clamp_min(0) / observed.sum((1, 2)).clamp_min(1)


@torch.no_grad()
def poisson_tv(
    counts,
    positions_m,
    live,
    *,
    regularization,
    vertical_strength=0,
    max_iterations=2000,
    tolerance=1e-4,
    check_interval=50,
):
    """Minimize Poisson fidelity plus anisotropic vertical/time TV, without targets.

    Uses Chambolle-Pock with theta=1 and tau*sigma*||K||^2<1. Vertical edges
    connect adjacent z locations on the same x/y detector string; no edge
    joins different strings or crosses a dead sensor. Diagnostics normalize
    primal-dual gaps per observed A photon. Output remains an intensity shape;
    probability normalization and population-prior shrinkage are separate.
    """
    if regularization < 0 or vertical_strength < 0 or max_iterations < 1 or tolerance <= 0:
        message = "Invalid Poisson-TV solver parameters"
        raise ValueError(message)
    observed = counts.double().masked_fill(~live[None, :, None], 0)
    if not torch.isfinite(observed).all() or (observed < 0).any():
        message = "Poisson-TV observations must be finite nonnegative photon counts"
        raise ValueError(message)
    if regularization == 0 or not (observed > 0).any():
        return observed, {"iterations": 0, "maximum_gap_per_photon": 0.0, "converged": True}
    order, inverse, weights = _physical_order(positions_m, live, vertical_strength)
    observed = observed[:, order]
    values = observed.clone()
    extrapolated = values.clone()
    temporal, vertical = tv_gradient(values * 0, weights)
    bound = 4 * (1 + (float(weights.max()) ** 2 if len(weights) else 0))
    step = 0.99 / math.sqrt(bound)
    support = live[order][None, :, None]
    maximum_gap = float("inf")
    for iteration in range(1, max_iterations + 1):
        dt, dz = tv_gradient(extrapolated, weights)
        temporal = (temporal + step * dt).clamp(-regularization, regularization)
        vertical = (vertical + step * dz).clamp(-regularization, regularization)
        descent = values - step * tv_adjoint(temporal, vertical, weights, values.shape)
        updated = poisson_prox(descent, observed, step).masked_fill(~support, 0)
        extrapolated = 2 * updated - values
        values = updated
        if iteration % check_interval == 0 or iteration == max_iterations:
            gap = poisson_tv_gap(values, observed, temporal, vertical, weights, regularization)
            maximum_gap = float(gap.max())
            if maximum_gap <= tolerance:
                break
    return values[:, inverse], {
        "iterations": iteration,
        "maximum_gap_per_photon": maximum_gap,
        "converged": maximum_gap <= tolerance,
    }
