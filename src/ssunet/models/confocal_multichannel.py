"""Joint-channel confocal models with separately normalized photon distributions."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from ..datasets.confocal import validate_patch_shape
from ..photon_methods import sinusoidal_photon_features
from ..softmax_fix import fixed_confocal_gap_loss, fixed_log_softmax
from .confocal_gap import ConfocalGAPUNet


class MultichannelConfocalGAP(nn.Module):
    """Use every retained detector channel as context and predict one PDF per channel.

    Inputs and outputs are [N,C,T,Z,Y,X]. Channels are optical detectors, not
    another convolved spatial axis. Each detector is split before inference;
    held-out photons from every detector stay out of the inputs.
    """

    def __init__(
        self,
        *,
        channels,
        dimensions=4,
        base_channels=32,
        depth=4,
        encoding_levels=10,
        backend="auto",
    ):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
            message = "The detector-channel count must be a positive integer"
            raise ValueError(message)
        self.channels, self.dimensions, self.encoding_levels = channels, dimensions, encoding_levels
        self.network = ConfocalGAPUNet(
            dimensions=dimensions,
            base_channels=base_channels,
            depth=depth,
            encoding_levels=encoding_levels,
            backend=backend,
        )
        self.channel_widths = self.network.channel_widths
        if channels > 1:
            convolution = nn.Conv3d
            kwargs = {}
            if dimensions == 4:
                from torch_convnd import Conv4d

                convolution = Conv4d
                kwargs = {"backend": backend}
            self.network.stem = convolution(channels * encoding_levels, base_channels, 1, **kwargs)
            self.network.head = convolution(base_channels, channels, 1, **kwargs)

    def forward(self, counts):
        """Jointly filter all detectors and crop before their separate normalization."""
        if (
            counts.ndim != 6
            or counts.shape[1] != self.channels
            or any(size < 1 for size in counts.shape)
        ):
            message = "Expected [N,C,T,Z,Y,X] with every configured detector channel"
            raise ValueError(message)
        batch, _, times, z, y, x = counts.shape
        padding = [
            p for size in reversed((z, y, x)) for p in (0, (-size) % self.network.pooling_multiple)
        ]
        values = functional.pad(counts.float(), padding)
        encoded = (
            sinusoidal_photon_features(values, self.encoding_levels).movedim(-1, 2).flatten(1, 2)
        )
        if self.dimensions == 3:
            encoded = encoded.movedim(2, 1).reshape(
                batch * times, self.channels * self.encoding_levels, *values.shape[-3:]
            )
        features = self.network.activation(self.network.stem(encoded.contiguous()))
        skips = []
        for level, block in enumerate(self.network.encoder):
            features = block(features)
            if level < len(self.network.down):
                skips.append(features)
                features = self.network.down[level](features)
        for projection, block, skip in zip(
            self.network.up_projection, self.network.decoder, reversed(skips), strict=True
        ):
            for axis in (-3, -2, -1):
                features = features.repeat_interleave(2, dim=axis)
            features = block(projection(features) + skip)
        logits = self.network.head(features)
        if self.dimensions == 3:
            logits = logits.reshape(batch, times, self.channels, *logits.shape[-3:]).movedim(2, 1)
        return logits[..., :z, :y, :x]


def channel_log_probabilities(logits):
    """Normalize TZYX per detector without coupling their probability masses."""
    if logits.ndim != 6:
        message = "Expected [N,C,T,Z,Y,X] logits"
        raise ValueError(message)
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    return fixed_log_softmax(work.flatten(2), dim=-1).reshape_as(logits)


def multichannel_gap_loss(logits, target, inputs):
    """Give each detector its own raw-count GAP target normalization and equal weight."""
    if logits.shape != target.shape or inputs.shape != target.shape or logits.ndim != 6:
        message = "Expected matching all-channel [N,C,T,Z,Y,X] tensors"
        raise ValueError(message)
    shape = (-1, 1, *logits.shape[2:])
    return fixed_confocal_gap_loss(
        logits.reshape(shape), target.reshape(shape), inputs.reshape(shape)
    )


def channel_expected_counts(logits, observed):
    """Calibrate each detector PDF with its own observed photon total."""
    if (
        logits.shape != observed.shape
        or observed.dtype.is_floating_point
        or observed.is_complex()
        or observed.dtype == torch.bool
        or (observed < 0).any()
    ):
        message = "Pass matching logits and raw nonnegative integer observations"
        raise ValueError(message)
    q = channel_log_probabilities(logits).exp()
    totals = observed.flatten(2).sum(-1)
    return q * totals[:, :, None, None, None, None]


def split_all_channels(counts, patch_tzyx, *, seed, index, batch_size=1, p=0.5):
    """Crop registered channels together and thin every detector independently."""
    counts = np.asarray(counts)
    validate_patch_shape(patch_tzyx)
    if (
        counts.ndim != 5
        or counts.shape[0] < 1
        or counts.dtype.kind not in "ui"
        or (counts.dtype.kind == "i" and np.any(counts < 0))
        or len(patch_tzyx) != 4
        or any(
            size < 1 or size > have for size, have in zip(patch_tzyx, counts.shape[1:], strict=True)
        )
        or not 0 < p < 1
        or batch_size < 1
    ):
        message = "Expected integer CTZYX counts, a fitting patch, and valid splitting options"
        raise ValueError(message)
    generator = np.random.default_rng(np.random.SeedSequence([seed, index]))
    patches = []
    for _ in range(batch_size):
        starts = [
            int(generator.integers(have - size + 1))
            for have, size in zip(counts.shape[1:], patch_tzyx, strict=True)
        ]
        selection = (
            slice(None),
            *(slice(start, start + size) for start, size in zip(starts, patch_tzyx, strict=True)),
        )
        patches.append(np.asarray(counts[selection], dtype=np.int64))
    full = np.stack(patches)
    retained = generator.binomial(full, p)
    return torch.from_numpy(retained), torch.from_numpy(full - retained)
