"""Select photon supervision from the acquisition format, following GAP/Bit2Bit."""

import torch

from .distributions import logits_to_log_probabilities
from .photon_reference import photon_count_log_probabilities


def method_for_observation(observation_kind: str) -> str:
    """Use the acquisition contract, not accidental sparsity in one sampled batch."""
    if observation_kind == "one_bit":
        return "bit2bit"
    if observation_kind == "photon_counts":
        return "gap"
    message = "Declare observation_kind as 'one_bit' or 'photon_counts'"
    raise ValueError(message)


def sinusoidal_photon_features(counts, levels=10, factor=10.0):
    """Use GAP's multiscale sine encoding, with ten levels by default."""
    if levels < 1 or factor <= 1:
        message = "Photon encoding needs positive levels and a factor greater than one"
        raise ValueError(message)
    return torch.stack([torch.sin(counts.float() * factor ** (-i)) for i in range(levels)], -1)


def paper_photon_loss(logits, target, live, inputs, *, observation_kind, epsilon=1e-8):
    """Use per-event unit-mean targets and method-specific normalization support.

    GAP counts use the log-mean-exp photon loss with mean-normalized targets.
    Binary Bit2Bit excludes input-hit entries from the normalization offset.
    Physical live support is separate from that binary dependence mask.
    """
    method = method_for_observation(observation_kind)
    if epsilon <= 0 or logits.shape != target.shape or inputs.shape != target.shape:
        message = "Expected matching photon tensors and a positive stabilization epsilon"
        raise ValueError(message)
    if target.is_floating_point() or (target < 0).any():
        message = "Pass raw nonnegative integer targets; normalization belongs in the loss"
        raise ValueError(message)
    if live.ndim == 1:
        live = live[None].expand(len(logits), -1)
    if live.dtype != torch.bool or live.shape != logits.shape[:2] or not live.any(1).all():
        message = "Each observation needs boolean live support matching the sensor axis"
        raise ValueError(message)
    if torch.any((target > 0) & ~live[..., None]):
        message = "Target photons lie outside live support"
        raise ValueError(message)
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    if method == "bit2bit":
        binary = ((inputs == 0) | (inputs == 1)).all() and ((target == 0) | (target == 1)).all()
        if not binary or torch.any((inputs > 0) & (target > 0)):
            message = "Bit2Bit requires disjoint splits of genuine one-bit observations"
            raise ValueError(message)
    locations = live.sum(1).to(work.dtype) * logits.shape[-1]
    totals = target.sum((1, 2)).to(work.dtype)
    weights = target.to(work.dtype) / (totals + epsilon * locations)[:, None, None]
    weight_sum = weights.sum((1, 2))
    if method == "gap":
        logq = photon_count_log_probabilities(work, live).masked_fill(target == 0, 0)
        return (-(weights * logq).sum((1, 2)) - weight_sum * locations.log()).mean()
    # Mask the exponential sum itself: zeroing a logit would leave exp(0)=1.
    # Stable log normalization needs no additive energy floor, which would
    # reintroduce an arbitrary absolute-logit anchor into this offset.
    support = live[..., None] & (inputs == 0)
    logq = logits_to_log_probabilities(work, dim=(1, 2), mask=support)
    logq = logq.masked_fill(target == 0, 0)
    return (-(weights * logq).sum((1, 2)) - weight_sum * locations.log()).mean()
