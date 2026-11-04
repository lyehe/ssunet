"""A small dense, sensor-permutation-equivariant photon reference backbone."""

import torch
import torch.nn.functional as functional
from torch import nn


class DensePhotonReference(nn.Module):
    """Share temporal kernels per sensor and mix events through pooled context."""

    def __init__(self, channels: int = 16):
        """Construct ordinary dense convolutions without sensor-index adjacency."""
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(1, channels, 7, padding=3),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 7, padding=3),
            nn.SiLU(),
        )
        self.sensor_context = nn.Sequential(nn.Linear(channels + 6, channels * 2), nn.SiLU())
        self.decoder = nn.Sequential(
            nn.Conv1d(channels * 3 + 7, channels, 1),
            nn.SiLU(),
            nn.Conv1d(channels, 1, 1),
        )

    def forward(
        self,
        counts: torch.Tensor,
        positions_m: torch.Tensor,
        live: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Return [batch,sensor,time] logits using observations and geometry only."""
        batch, sensors, bins = counts.shape
        if live.ndim == 1:
            live = live.unsqueeze(0).expand(batch, -1)
        if positions_m.ndim == 2:
            positions_m = positions_m.unsqueeze(0).expand(batch, -1, -1)
        values = counts.float().masked_fill(~live[..., None], 0)
        features = self.temporal(values.log1p().reshape(batch * sensors, 1, bins))
        coarse_bins = max(1, bins // 4)
        features = functional.adaptive_avg_pool1d(features, coarse_bins)
        features = features.reshape(batch, sensors, -1, coarse_bins)
        coordinates = positions_m.float() / 500.0
        local_count = values.sum(dim=-1).log1p()[..., None]
        total_count = values.sum(dim=(1, 2)).log1p()[:, None, None].expand(-1, sensors, -1)
        fraction = p.reshape(batch, 1, 1).expand(-1, sensors, -1)
        observed = torch.cat([coordinates, local_count, total_count, fraction], dim=-1)
        sensor_features = self.sensor_context(torch.cat([features.mean(dim=-1), observed], dim=-1))
        context = (sensor_features * live[..., None]).sum(dim=1) / live.sum(dim=1, keepdim=True)
        context = context[:, None, :, None].expand(-1, sensors, -1, coarse_bins)
        time_coordinate = torch.linspace(-1, 1, coarse_bins, device=counts.device)
        time_coordinate = time_coordinate.reshape(1, 1, 1, -1).expand(batch, sensors, -1, -1)
        decoder_input = torch.cat(
            [
                features,
                context,
                observed[..., None].expand(-1, -1, -1, coarse_bins),
                time_coordinate,
            ],
            dim=2,
        )
        output = self.decoder(decoder_input.reshape(batch * sensors, -1, coarse_bins))
        output = functional.interpolate(output, size=bins, mode="linear", align_corners=False)
        return output.reshape(batch, sensors, bins)
