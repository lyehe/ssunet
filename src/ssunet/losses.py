"""Loss functions for relative photon-location predictions."""

import math

import torch

from .constants import EPSILON
from .distributions import _distribution_dims, logits_to_log_probabilities


def mse_loss(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the mean squared error loss.

    This function calculates the mean squared error (MSE) loss between the predicted result
    and the target. The MSE is defined as the average of the squared differences between
    the predicted and actual values.

    The function applies the following steps:
    1. If a mask is provided, both result and target are multiplied by the mask.
    2. The result is exponentiated (exp_energy = exp(result)).
    3. Both exp_energy and target are normalized by their respective means.
    4. The MSE is computed as the mean of (exp_energy - target)^2.

    Note: The normalization step may affect the interpretation of the MSE as a standard
    error metric. This implementation appears to be a custom variation of MSE.
    """
    result = result * mask if mask is not None else result
    target = target * mask if mask is not None else target
    exp_energy = torch.exp(result)
    exp_energy = exp_energy / (torch.mean(exp_energy, dim=(-1, -2, -3, -4), keepdim=True) + EPSILON)
    target = target / (torch.mean(target, dim=(-1, -2, -3, -4), keepdim=True) + EPSILON)
    return torch.mean((exp_energy - target) ** 2)


def l1_loss(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the mean absolute error (L1) loss.

    This function calculates the L1 loss between the predicted result and the target.
    The L1 loss is defined as the average of the absolute differences between
    the predicted and actual values.

    The function applies the following steps:
    1. If a mask is provided, both result and target are multiplied by the mask.
    2. The result is exponentiated (exp_energy = exp(result)).
    3. Both exp_energy and target are normalized by their respective means.
    4. The L1 loss is computed as the mean of |exp_energy - target|.
    """
    result = result * mask if mask is not None else result
    target = target * mask if mask is not None else target
    exp_energy = torch.exp(result)
    exp_energy = exp_energy / (torch.mean(exp_energy, dim=(-1, -2, -3, -4), keepdim=True) + EPSILON)
    target = target / (torch.mean(target, dim=(-1, -2, -3, -4), keepdim=True) + EPSILON)
    return torch.mean(torch.abs(exp_energy - target))


def photon_loss(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Predict held-out photon locations from 3D network log weights.

    Targets are photon observations or their normalized event weights; clean
    pixel targets are not required. Normalization covers channels, time and
    both spatial axes independently for each batch item. An optional Bit2Bit
    mask excludes input-photon locations from both target scoring and the
    exponential normalization sum. Masked locations have zero gradients and
    contribute zero normalization mass. The stable calculation keeps the
    objective's mean reduction and additive log-size convention.
    """
    return _photon_loss(result, target, mask, dim=(-1, -2, -3, -4))


def photon_loss_2d(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the photon loss for 2D data.

    Similar to the 3D version, but for use with 2D pairs instead of 3D volumes.
    """
    return _photon_loss(result, target, mask, dim=(-1, -2, -4))


def _masked_photon_weights(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Promote precision and mask targets without anchoring excluded logits at zero."""
    result = result.float() if result.dtype in (torch.float16, torch.bfloat16) else result
    target = target.to(dtype=result.dtype)
    if mask is not None:
        target = target * mask
    return result, target


def _photon_loss(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    dim: tuple[int, ...],
) -> torch.Tensor:
    """Evaluate log-mean-exp with the Bit2Bit mask inside its normalization offset."""
    result, target = _masked_photon_weights(result, target, mask)
    dims = _distribution_dims(result, dim)
    support = mask.bool() if mask is not None else None
    log_probability = logits_to_log_probabilities(result, dim=dims, mask=support)
    log_probability = log_probability.masked_fill(target == 0, 0)
    # The historical mean and log-size convention still use the original
    # lattice size; the exponential sum contains only unmasked locations.
    log_locations = math.log(math.prod(result.shape[axis] for axis in dims))
    return torch.mean(-target * (log_probability + log_locations))


def photon_nll(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    dim: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Compute target-event-normalized photon cross entropy for validation.

    Empty target distributions contribute no score. When ``mask`` is supplied,
    this is a masked Bit2Bit score rather than a likelihood on independent
    observations. This diagnostic is separate from the historical training
    loss, which has a different additive offset and event weighting.
    """
    result, target = _masked_photon_weights(result, target, mask)
    dims = _distribution_dims(result, dim)
    support = mask.bool() if mask is not None else None
    log_probability = logits_to_log_probabilities(result, dim=dims, mask=support)
    log_probability = log_probability.masked_fill(target == 0, 0)
    counts = target.sum(dim=dims, keepdim=True)
    valid = counts > 0
    safe_counts = torch.where(valid, counts, torch.ones_like(counts))
    per_distribution = -(target * log_probability).sum(dim=dims, keepdim=True) / safe_counts
    return (per_distribution * valid).sum() / valid.sum().clamp_min(1)


loss_functions = {
    "mse": mse_loss,
    "l1": l1_loss,
    "photon": photon_loss,
    "photon_2d": photon_loss_2d,
}
