"""Conversions from network log weights to photon-location distributions."""

import torch


def _distribution_dims(logits: torch.Tensor, dim: int | tuple[int, ...] | None) -> tuple[int, ...]:
    """Resolve the axes of one distribution, retaining batch axes by default."""
    if dim is None:
        dims = tuple(range(1, logits.ndim))
    elif isinstance(dim, int):
        dims = (dim,)
    else:
        dims = tuple(dim)
    if not dims:
        message = "A photon distribution must have at least one normalization axis"
        raise ValueError(message)
    if any(axis < -logits.ndim or axis >= logits.ndim for axis in dims):
        message = "Photon distribution normalization axis is out of range"
        raise ValueError(message)
    dims = tuple(axis % logits.ndim for axis in dims)
    if len(set(dims)) != len(dims):
        message = "Photon distribution normalization axes must be distinct"
        raise ValueError(message)
    return dims


def logits_to_log_probabilities(
    logits: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize photon log weights over the selected axes.

    By default all non-batch axes form one distribution. Half and bfloat16
    inputs are accumulated in float32; double precision inputs retain their
    precision. The network's ``forward`` output remains unrestricted logits.
    A boolean mask excludes entries from the exponential normalization sum.
    An entirely masked distribution returns negative infinity everywhere;
    it carries no probability mass and must have no scored target events.
    """
    if not logits.is_floating_point():
        message = "Photon logits must be a floating point tensor"
        raise TypeError(message)
    dims = _distribution_dims(logits, dim)
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    if mask is not None:
        if mask.dtype != torch.bool:
            message = "Photon probability support must be a boolean mask"
            raise TypeError(message)
        support = torch.broadcast_to(mask, work.shape)
        has_support = support.any(dim=dims, keepdim=True)
        work = work.masked_fill(~support, -torch.inf)
        # A finite dummy row avoids undefined all-minus-infinity gradients.
        # Its entries are all excluded again after the stable normalization.
        work = work.masked_fill(~has_support, 0)
    centered = work - work.amax(dim=dims, keepdim=True)
    log_probability = centered - torch.logsumexp(centered, dim=dims, keepdim=True)
    return (
        log_probability.masked_fill(~support, -torch.inf) if mask is not None else log_probability
    )


def logits_to_probabilities(
    logits: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return normalized photon-location probabilities from network logits.

    The result sums to one over ``dim`` wherever the optional boolean mask
    leaves support, and to zero for fully masked distributions. It is unchanged
    by a common additive logit offset. Count-scaled intensity estimates can be
    formed separately by multiplying by the observed photon count.
    """
    return logits_to_log_probabilities(logits, dim=dim, mask=mask).exp()
