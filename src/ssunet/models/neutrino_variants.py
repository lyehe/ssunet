"""Finite mechanism-study variants that preserve observation-only conditioning."""

import torch
from torch import nn

from .neutrino_dense import DensePhotonReference


class FinePhotonReference(DensePhotonReference):
    """Keep all temporal bins while sharing the unchanged parameter layout."""

    def forward(self, counts, positions_m, live, p):
        """Decode full-resolution temporal features without the coarse bottleneck."""
        batch, sensors, bins = counts.shape
        if live.ndim == 1:
            live = live.unsqueeze(0).expand(batch, -1)
        if positions_m.ndim == 2:
            positions_m = positions_m.unsqueeze(0).expand(batch, -1, -1)
        values = counts.float().masked_fill(~live[..., None], 0)
        features = self.temporal(values.log1p().reshape(batch * sensors, 1, bins))
        features = features.reshape(batch, sensors, -1, bins)
        coordinates = positions_m.float() / 500.0
        local_count = values.sum(-1).log1p()[..., None]
        total_count = values.sum((1, 2)).log1p()[:, None, None].expand(-1, sensors, -1)
        fraction = p.reshape(batch, 1, 1).expand(-1, sensors, -1)
        observed = torch.cat([coordinates, local_count, total_count, fraction], -1)
        sensor_features = self.sensor_context(torch.cat([features.mean(-1), observed], -1))
        context = (sensor_features * live[..., None]).sum(1) / live.sum(1, keepdim=True)
        context = context[:, None, :, None].expand(-1, sensors, -1, bins)
        time = torch.linspace(-1, 1, bins, device=counts.device)
        time = time.reshape(1, 1, 1, -1).expand(batch, sensors, -1, -1)
        decoder_input = torch.cat(
            [features, context, observed[..., None].expand(-1, -1, -1, bins), time], 2
        )
        return self.decoder(decoder_input.reshape(batch * sensors, -1, bins)).reshape(
            batch, sensors, bins
        )


class PriorResidualPhotonReference(nn.Module):
    """Initialize exactly at a training-prior/count mixture, then learn corrections."""

    def __init__(self, prior, time_kernel, fraction=0.25, channels=16):
        """Store training-only references and a zero-initialized correction scale."""
        super().__init__()
        self.backbone = DensePhotonReference(channels)
        self.correction_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer("prior", torch.as_tensor(prior, dtype=torch.float32))
        self.register_buffer("time_kernel", torch.as_tensor(time_kernel, dtype=torch.float32))
        self.fraction = fraction

    def forward(self, counts, positions_m, live, p):
        """Return mixture log-probabilities plus an A-conditioned learned residual."""
        if live.ndim == 1:
            live = live[None].expand(len(counts), -1)
        prior = self.prior[None] * live[..., None]
        prior = prior / prior.sum((1, 2), keepdim=True)
        observed = counts.float().masked_fill(~live[..., None], 0) @ self.time_kernel
        total = observed.sum((1, 2), keepdim=True)
        empirical = torch.where(total > 0, observed / total.clamp_min(1), prior)
        mixture = (1 - self.fraction) * empirical + self.fraction * prior
        logits = self.backbone(counts, positions_m, live, p)
        return (
            mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
            + self.correction_scale * logits
        )
