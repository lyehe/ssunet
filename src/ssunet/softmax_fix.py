"""Photon normalization using Barron et al. (2026), equation 5.

Reference: https://jonbarron.info/softmaxfix/ . Excluding the selected maximum
from log-sum-exp and combining with softplus preserves tiny log probabilities
and the dominant entry's derivative. No global PyTorch functions are patched.
The separate teacher-logit distillation correction is outside this count loss.
"""

import math

import torch
from torch.nn import functional


def fixed_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize finite logits while preserving highly confident losses/gradients.

    Half/bfloat16 inputs are accumulated in float32. Double precision retains
    its precision. This finite-support implementation is used for complete
    confocal image patches; it does not replace Bit2Bit's masked offset.
    """
    if not logits.is_floating_point():
        message = "Log-softmax requires floating-point logits"
        raise TypeError(message)
    if logits.ndim == 0 or not -logits.ndim <= dim < logits.ndim:
        message = "Normalization axis is outside the logit tensor"
        raise ValueError(message)
    if logits.shape[dim] == 0 or not torch.isfinite(logits).all():
        message = "This fixed log-softmax requires nonempty, finite logit support"
        raise ValueError(message)
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    if work.shape[dim] == 1:
        return work * 0
    selected = work.argmax(dim=dim, keepdim=True)
    maximum = work.gather(dim, selected)
    shifted = (work - maximum).scatter(dim, selected, 0.0)
    remaining = shifted.scatter(dim, selected, -torch.inf)
    log_other_mass = torch.logsumexp(remaining, dim=dim, keepdim=True)
    return shifted - functional.softplus(log_other_mass)


def fixed_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exponentiate fixed log-softmax, retaining its stable backward computation."""
    return fixed_log_softmax(logits, dim=dim).exp()


def fixed_confocal_gap_loss(logits, target, inputs, *, epsilon=1e-8):
    """Keep GAP's count supervision and unit-mean target weighting unchanged."""
    if logits.shape != target.shape or inputs.shape != target.shape or logits.ndim != 6:
        message = "Expected matching [N,1,T,Z,Y,X] confocal tensors"
        raise ValueError(message)
    if (
        epsilon <= 0
        or target.is_floating_point()
        or target.is_complex()
        or target.dtype == torch.bool
        or torch.any(target < 0)
    ):
        message = "GAP targets must remain raw nonnegative integer counts"
        raise ValueError(message)
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    flat = work.flatten(1)
    counts = target.flatten(1).to(work.dtype)
    locations = flat.shape[1]
    weights = counts / (counts.sum(-1, keepdim=True) + epsilon * locations)
    logq = fixed_log_softmax(flat).masked_fill(counts == 0, 0)
    return (-(weights * logq).sum(-1) - weights.sum(-1) * math.log(locations)).mean()
