"""Controlled Poisson reference and physical-support photon count likelihood."""

from __future__ import annotations

import numpy as np
import torch

from .datasets.neutrino import split_photon_counts
from .distributions import logits_to_log_probabilities


def photon_count_log_probabilities(logits: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """Normalize on live sensors; inactive support has exactly zero probability."""
    if logits.ndim != 3 or live.dtype != torch.bool:
        message = "Expected [batch,sensor,time] logits and a boolean live mask"
        raise ValueError(message)
    if live.ndim == 1:
        live = live.unsqueeze(0).expand(logits.shape[0], -1)
    if live.shape != logits.shape[:2] or not live.any(dim=1).all():
        message = "Each observation must have at least one live sensor"
        raise ValueError(message)
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    masked = work.masked_fill(~live[..., None], -torch.inf)
    return logits_to_log_probabilities(masked)


def photon_count_nll(
    logits: torch.Tensor, target: torch.Tensor, live: torch.Tensor
) -> torch.Tensor:
    """Average the raw held-out photon NLL per event, retaining photon weighting."""
    if target.shape != logits.shape or target.is_floating_point() or target.dtype == torch.bool:
        message = "Photon targets must be integer counts with the same shape as logits"
        raise ValueError(message)
    if (target < 0).any():
        message = "Photon counts must be nonnegative"
        raise ValueError(message)
    if live.ndim == 1:
        live = live.unsqueeze(0).expand(logits.shape[0], -1)
    if torch.any((target > 0) & ~live[..., None]):
        message = "Target contains photons outside live support"
        raise ValueError(message)
    log_probability = photon_count_log_probabilities(logits, live)
    # Avoid 0*(-inf), while preserving zero gradients on invalid/empty targets.
    scored = log_probability.masked_fill(target == 0, 0)
    return -(target.to(scored.dtype) * scored).sum(dim=(1, 2)).mean()


def _max_moment_z_score(observed, expected, standard_error) -> float:
    """Return the largest moment deviation with nonzero sampling uncertainty."""
    active = standard_error > 0
    return float(np.max(np.abs(observed[active] - expected[active]) / standard_error[active]))


def audit_poisson_thinning(
    *, seed: int = 42, samples: int = 100000, sigma_limit: float = 6.0
) -> dict:
    """Check conditional moments on regenerated Poisson data at fixed intensities."""
    if samples < 1000 or not np.isfinite(sigma_limit) or sigma_limit <= 0:
        message = "Use at least 1000 realizations and a positive finite sigma limit"
        raise ValueError(message)
    rates = np.asarray([0.0, 0.05, 0.5, 2.0, 10.0, 30.0])
    generator = np.random.default_rng(seed)
    reports = []
    for p in (0.25, 0.5, 0.75):
        counts = generator.poisson(rates, size=(samples, len(rates)))
        retained, target = split_photon_counts(counts, p, generator)
        x_mean, y_mean = retained.mean(axis=0), target.mean(axis=0)
        x_variance, y_variance = retained.var(axis=0, ddof=1), target.var(axis=0, ddof=1)
        covariance = ((retained - x_mean) * (target - y_mean)).sum(axis=0) / (samples - 1)
        expected_x, expected_y = p * rates, (1 - p) * rates
        z_scores = {
            "retained_mean": _max_moment_z_score(x_mean, expected_x, np.sqrt(expected_x / samples)),
            "target_mean": _max_moment_z_score(y_mean, expected_y, np.sqrt(expected_y / samples)),
            "retained_variance": _max_moment_z_score(
                x_variance, expected_x, np.sqrt((expected_x + 2 * expected_x**2) / samples)
            ),
            "target_variance": _max_moment_z_score(
                y_variance, expected_y, np.sqrt((expected_y + 2 * expected_y**2) / samples)
            ),
            "conditional_covariance": _max_moment_z_score(
                covariance, np.zeros_like(rates), np.sqrt(expected_x * expected_y / samples)
            ),
        }
        zero_preserved = bool(np.all(retained[:, 0] == 0) and np.all(target[:, 0] == 0))
        reports.append(
            {
                "p": p,
                "rates": rates.tolist(),
                "retained_mean": x_mean.tolist(),
                "target_mean": y_mean.tolist(),
                "retained_variance": x_variance.tolist(),
                "target_variance": y_variance.tolist(),
                "conditional_covariance": covariance.tolist(),
                "z_scores": z_scores,
                "conserves_counts": bool(np.array_equal(retained + target, counts)),
                "zero_rate_preserved": zero_preserved,
                "multiphoton_bins_present": bool(np.max(counts) > 1),
                "passed": zero_preserved and max(z_scores.values()) < sigma_limit,
            }
        )
    frozen = np.full(samples, 10, dtype=np.int64)
    left, right = split_photon_counts(frozen, 0.5, generator)
    frozen_covariance = float(np.cov(left, right)[0, 1])
    return {
        "reference_kind": "analytic_independent_poisson_counts_not_physical_simulation",
        "conditioning": "fixed underlying intensity; full observation regenerated each realization",
        "samples_per_rate": samples,
        "sigma_limit": sigma_limit,
        "cases": reports,
        "fixed_total_negative_control": {
            "total": 10,
            "p": 0.5,
            "observed_covariance": frozen_covariance,
            "expected_covariance": -2.5,
        },
        "passed": all(item["passed"] for item in reports) and frozen_covariance < 0,
    }
