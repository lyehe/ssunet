"""Photon-count scaling and temporal alignment shared by inference entry points."""

from collections.abc import Callable, Iterable

import numpy as np
import torch

from .distributions import logits_to_probabilities


def temporal_window_size(model: torch.nn.Module, requested: int) -> int:
    """Round context depth up to the network's temporal downsampling factor."""
    if requested <= 0:
        message = "Window size must be positive"
        raise ValueError(message)
    config = getattr(model, "config", None)
    depth = max(0, getattr(config, "depth", 1) - 1)
    stages = max(0, getattr(config, "z_conv_stage", 0))
    factor = 2 ** min(depth, stages)
    return ((requested + factor - 1) // factor) * factor


def model_distribution_dims(model: torch.nn.Module) -> tuple[int, ...] | None:
    """Return normalization axes corresponding to the model's photon loss."""
    config = getattr(model, "config", None)
    return (-1, -2, -4) if getattr(config, "loss_function", None) == "photon_2d" else None


def logits_to_expected_counts(
    logits: torch.Tensor,
    observed: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Scale each predicted photon-location distribution by its observed count.

    Both tensors use batch-first layout. The output has the same units as the
    observations; an empty observation produces zeros without a division by zero.
    Counts and probabilities are reduced over the same ``dim`` axes, allowing
    per-frame distributions for the 2D photon objective.
    """
    if logits.shape != observed.shape:
        message = "Predictions and observations must have matching shapes"
        raise ValueError(message)
    probabilities = logits_to_probabilities(logits, dim=dim)
    dims = tuple(range(1, observed.ndim)) if dim is None else dim
    counts = observed.to(probabilities.dtype).sum(dim=dims, keepdim=True)
    return probabilities * counts


def reflected_window(data: np.ndarray, start: int, size: int) -> np.ndarray:
    """Return a temporal window, reflecting indices outside the recorded frames."""
    if data.ndim < 1 or data.shape[0] == 0:
        message = "A temporal window requires at least one recorded frame"
        raise ValueError(message)
    if size <= 0:
        message = "Window size must be positive"
        raise ValueError(message)
    indices = np.arange(start, start + size)
    if data.shape[0] == 1:
        indices = np.zeros(size, dtype=np.int64)
    else:
        period = 2 * (data.shape[0] - 1)
        indices %= period
        indices = np.where(indices < data.shape[0], indices, period - indices)
    return data[indices]


def average_temporal_offsets(
    data: np.ndarray,
    predict_window: Callable[[np.ndarray], np.ndarray],
    window_size: int = 32,
    offsets: Iterable[int] = range(-8, 8),
    on_offset: Callable[[int, np.ndarray], None] | None = None,
    on_window: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average predictions of the same frames from different temporal contexts.

    ``data`` and each prediction have layout ``(time, height, width, channel)``.
    The predictor must return expected counts in the observation's units, rather
    than unnormalized exponentiated logits. Each offset tiles the original time
    axis; reflected padding provides context and never becomes an output frame.
    ``on_offset`` can persist each aligned volume. Coverage counts are returned
    alongside the averaged volume so callers can inspect boundary coverage.
    """
    if data.ndim != 4 or any(size == 0 for size in data.shape):
        message = "Expected a nonempty (time, height, width, channel) volume"
        raise ValueError(message)
    if window_size <= 0:
        message = "Window size must be positive"
        raise ValueError(message)
    offsets = tuple(offsets)
    if not offsets:
        message = "At least one temporal offset is required"
        raise ValueError(message)
    if any(not isinstance(offset, (int, np.integer)) for offset in offsets):
        message = "Temporal offsets must be integers"
        raise ValueError(message)

    length = data.shape[0]
    accumulated = np.zeros(data.shape, dtype=np.float64)
    coverage = np.zeros(length, dtype=np.int32)
    for offset in offsets:
        # Include a leading partial tile for positive offsets, and a trailing
        # partial tile whenever needed. Every real frame is covered per offset.
        first_start = offset + ((-offset) // window_size) * window_size
        volume = np.zeros(data.shape, dtype=np.float32)
        offset_coverage = np.zeros(length, dtype=np.int32)
        for start in range(first_start, length, window_size):
            window = reflected_window(data, start, window_size)
            prediction = np.asarray(predict_window(window))
            if prediction.shape != window.shape:
                message = "Window prediction must match the input window shape"
                raise ValueError(message)
            begin = max(0, start)
            end = min(length, start + window_size)
            local_begin = begin - start
            volume[begin:end] += prediction[local_begin : local_begin + end - begin]
            offset_coverage[begin:end] += 1
            if on_window is not None:
                on_window(int(offset), start)

        volume /= offset_coverage[:, None, None, None]
        accumulated += volume
        coverage += offset_coverage > 0
        if on_offset is not None:
            on_offset(int(offset), volume)

    averaged = accumulated / coverage[:, None, None, None]
    return averaged.astype(np.float32), coverage
