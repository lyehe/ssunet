"""Compact XYZ and XYZT U-Nets for GAP supervision of confocal photon counts."""

from functools import partial
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional

from ..photon_methods import sinusoidal_photon_features


class _ResidualBlock(nn.Module):
    """Refine features with two unrestricted kernels and an additive shortcut."""

    def __init__(self, convolution, channels):
        super().__init__()
        self.layers = nn.Sequential(
            convolution(channels, channels, 3, padding=1),
            nn.SiLU(),
            convolution(channels, channels, 3, padding=1),
        )
        self.activation = nn.SiLU()

    def forward(self, features):
        """Keep the shortcut and output at the same resolution."""
        return self.activation(features + self.layers(features))


class ConfocalGAPUNet(nn.Module):
    """Predict full-resolution logits from one count channel.

    Both modes accept and return ``[N, 1, T, Z, Y, X]``. The 3D control processes
    each time point independently. The 4D mode uses full 3x3x3x3 kernels and
    preserves time resolution throughout the U-Net. Only XYZ are downsampled.
    Targets and counts from another detector channel never enter the network.
    """

    def __init__(
        self, *, dimensions=4, base_channels=12, depth=3, encoding_levels=10, backend="auto"
    ):
        super().__init__()
        if dimensions not in (3, 4) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < minimum
            for value, minimum in ((base_channels, 1), (depth, 2), (encoding_levels, 1))
        ):
            message = (
                "Use dimensions=3 or 4, integer depth >= 2, and positive channel/encoding counts"
            )
            raise ValueError(message)
        if dimensions == 4:
            from torch_convnd import Conv4d

            convolution = partial(Conv4d, backend=backend)
            down_kernel = (1, 2, 2, 2)
        else:
            convolution = nn.Conv3d
            down_kernel = (2, 2, 2)
        self.dimensions = dimensions
        self.encoding_levels = encoding_levels
        self.pooling_multiple = 2 ** (depth - 1)
        widths = tuple(base_channels * 2**level for level in range(depth))
        self.channel_widths = widths
        self.stem = convolution(encoding_levels, widths[0], 1)
        self.encoder = nn.ModuleList(_ResidualBlock(convolution, width) for width in widths)
        self.down = nn.ModuleList(
            convolution(left, right, down_kernel, stride=down_kernel)
            for left, right in pairwise(widths)
        )
        self.up_projection = nn.ModuleList(
            convolution(right, left, 1) for left, right in reversed(tuple(pairwise(widths)))
        )
        self.decoder = nn.ModuleList(
            _ResidualBlock(convolution, width) for width in reversed(widths[:-1])
        )
        self.head = convolution(widths[0], 1, 1)
        self.activation = nn.SiLU()

    def forward(self, counts):
        """Encode raw retained counts and crop padding before probability normalization."""
        if counts.ndim != 6 or counts.shape[1] != 1 or any(s < 1 for s in counts.shape):
            message = "Confocal counts must have shape [N,1,T,Z,Y,X] with nonempty axes"
            raise ValueError(message)
        batch, _, times, z_size, y_size, x_size = counts.shape
        spatial = counts.shape[-3:]
        pad = [
            value for size in reversed(spatial) for value in (0, (-size) % self.pooling_multiple)
        ]
        values = functional.pad(counts[:, 0].float(), pad)
        encoded = sinusoidal_photon_features(values, self.encoding_levels)
        if self.dimensions == 3:
            # Moving T into the batch prevents convolution or normalization across T.
            encoded = encoded.reshape(batch * times, *values.shape[-3:], self.encoding_levels)
        features = self.activation(self.stem(encoded.movedim(-1, 1).contiguous()))
        skips = []
        for level, block in enumerate(self.encoder):
            features = block(features)
            if level < len(self.down):
                skips.append(features)
                features = self.down[level](features)
        for projection, block, skip in zip(
            self.up_projection, self.decoder, reversed(skips), strict=True
        ):
            for axis in (-3, -2, -1):
                features = features.repeat_interleave(2, dim=axis)
            features = block(projection(features) + skip)
        logits = self.head(features)
        if self.dimensions == 3:
            logits = logits.reshape(batch, times, *logits.shape[-3:])[:, None]
        return logits[..., :z_size, :y_size, :x_size]


def confocal_gap_loss(logits, target, inputs):
    """Reuse the verified GAP loss over the complete TZYX patch support."""
    from ..photon_methods import paper_photon_loss

    if logits.shape != target.shape or inputs.shape != target.shape or logits.ndim != 6:
        message = "Confocal inputs, targets and logits must have matching [N,1,T,Z,Y,X] shapes"
        raise ValueError(message)
    batch = len(logits)
    live = torch.ones(batch, 1, device=logits.device, dtype=torch.bool)
    return paper_photon_loss(
        logits.reshape(batch, 1, -1),
        target.reshape(batch, 1, -1),
        live,
        inputs.reshape(batch, 1, -1),
        observation_kind="photon_counts",
    )
