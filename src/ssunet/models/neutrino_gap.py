"""Dense photon architectures with GAP's published raw-count sine encoding."""

import math

import torch
from torch import nn
from torch.nn import functional as functional
from torch_convnd import Conv4d

from ..photon_methods import sinusoidal_photon_features
from .neutrino_conv4d import GridPhotonReference4d
from .neutrino_dense import DensePhotonReference


def observed_features(counts, positions_m, live, p):
    """Keep geometry and observed brightness as input-only auxiliary features."""
    batch, sensors, _ = counts.shape
    if live.ndim == 1:
        live = live[None].expand(batch, -1)
    if positions_m.ndim == 2:
        positions_m = positions_m[None].expand(batch, -1, -1)
    values = counts.float().masked_fill(~live[..., None], 0)
    local = values.sum(-1).log1p()[..., None]
    total = values.sum((1, 2)).log1p()[:, None, None].expand(-1, sensors, -1)
    fraction = p.reshape(batch, 1, 1).expand(-1, sensors, -1)
    metadata = torch.cat([positions_m.float() / 500.0, local, total, fraction], -1)
    return values, metadata, positions_m, live


class GAPTemporalPhotonReference(DensePhotonReference):
    """Compare coarse and full temporal decoding using identical GAP input features."""

    def __init__(self, channels=16, *, full_time=False, encoding_levels=10):
        """Retain the reference decoder/context and replace only the count embedding."""
        super().__init__(channels)
        self.full_time = full_time
        self.encoding_levels = encoding_levels
        self.temporal[0] = nn.Conv1d(encoding_levels, channels, 7, padding=3)

    def forward(self, counts, positions_m, live, p):
        """Return observation-conditioned logits, without target-based input scaling."""
        values, observed, _, live = observed_features(counts, positions_m, live, p)
        batch, sensors, bins = values.shape
        encoded = sinusoidal_photon_features(values, self.encoding_levels)
        features = self.temporal(
            encoded.permute(0, 1, 3, 2).reshape(batch * sensors, self.encoding_levels, bins)
        )
        decoded_bins = bins if self.full_time else max(1, bins // 4)
        features = functional.adaptive_avg_pool1d(features, decoded_bins)
        features = features.reshape(batch, sensors, -1, decoded_bins)
        local = self.sensor_context(torch.cat([features.mean(-1), observed], -1))
        context = (local * live[..., None]).sum(1) / live.sum(1, keepdim=True)
        context = context[:, None, :, None].expand(-1, sensors, -1, decoded_bins)
        time = torch.linspace(-1, 1, decoded_bins, device=counts.device)
        time = time.reshape(1, 1, 1, -1).expand(batch, sensors, -1, -1)
        decoded = torch.cat(
            [features, context, observed[..., None].expand(-1, -1, -1, decoded_bins), time], 2
        )
        logits = self.decoder(decoded.reshape(batch * sensors, -1, decoded_bins))
        logits = functional.interpolate(logits, size=bins, mode="linear", align_corners=False)
        return logits.reshape(batch, sensors, bins)


class GAPGridPhotonReference4d(GridPhotonReference4d):
    """Use full x/y/z/time kernels with the same sine encoding as both controls."""

    def __init__(self, positions_m, channels=4, backend="auto", encoding_levels=10):
        """Reuse the validated lattice mapping and full 4D residual blocks."""
        super().__init__(positions_m, channels, backend)
        self.encoding_levels = encoding_levels
        self.input_features = encoding_levels + 8
        self.stem = Conv4d(self.input_features, channels, 1, backend=backend)

    def forward(self, counts, positions_m, live, p):
        """Gather logits at actual detectors before live-support normalization."""
        values, observed, positions_m, live = observed_features(counts, positions_m, live, p)
        batch, sensors, bins = counts.shape
        index = self.sensor_indices(positions_m)
        metadata = torch.cat([live[..., None], observed], -1)
        time = torch.linspace(-1, 1, bins, device=counts.device).expand(batch, sensors, -1)
        features = torch.cat(
            [
                sinusoidal_photon_features(values, self.encoding_levels),
                metadata[:, :, None].expand(-1, -1, bins, -1),
                time[..., None],
            ],
            -1,
        )
        features = features.masked_fill(~live[:, :, None, None], 0).permute(0, 3, 1, 2)
        cells = math.prod(self.grid_shape)
        grid = features.new_zeros(batch, self.input_features, cells, bins).scatter(
            2, index[:, None, :, None].expand(-1, self.input_features, -1, bins), features
        )
        hidden = self.activation(
            self.stem(grid.reshape(batch, self.input_features, *self.grid_shape, bins))
        )
        for block in self.blocks:
            hidden = hidden + self.activation(block(hidden))
        logits = self.head(hidden).reshape(batch, cells, bins)
        return logits.gather(1, index[..., None].expand(-1, -1, bins))
