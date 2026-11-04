"""Dense marked-pulse distributions without graph construction or sensor kernels."""

import torch
import torch.nn.functional as functional
from torch import nn


class DenseMarkedPulseReference(nn.Module):
    """Share temporal convolutions across fixed spatial cells with charge-mark channels."""

    def __init__(self, marks=4, channels=16):
        """Keep mark channels internal, avoiding one encoder for every virtual DOM."""
        super().__init__()
        self.marks = marks
        self.temporal = nn.Sequential(
            nn.Conv1d(marks, channels, 7, padding=3),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 7, padding=3),
            nn.SiLU(),
        )
        self.sensor_context = nn.Sequential(nn.Linear(channels + 6, 2 * channels), nn.SiLU())
        self.decoder = nn.Sequential(
            nn.Conv1d(channels * 3 + 7, channels, 1), nn.SiLU(), nn.Conv1d(channels, marks, 1)
        )

    def forward(self, counts, positions_m, live, p):
        """Predict a normalized joint DOM/time/mark distribution from A only."""
        with torch.autocast(device_type=counts.device.type, enabled=False):
            return self._forward_float32(counts, positions_m, live, p)

    def _forward_float32(self, counts, positions_m, live, p):
        """Preserve projection precision on the larger marked-pulse support."""
        batch, virtual_sensors, bins = counts.shape
        sensors = virtual_sensors // self.marks
        if virtual_sensors % self.marks:
            message = "Marked pulse tensor has an incomplete DOM/mark block"
            raise ValueError(message)
        if live.ndim == 1:
            live = live[None].expand(batch, -1)
        if positions_m.ndim == 2:
            positions_m = positions_m[None].expand(batch, -1, -1)
        dom_live = live.reshape(batch, sensors, self.marks).all(2)
        coordinates = positions_m.reshape(batch, sensors, self.marks, 3)[:, :, 0] / 500.0
        marked = (
            counts.float()
            .masked_fill(~live[..., None], 0)
            .reshape(batch, sensors, self.marks, bins)
        )
        values = marked.sum(2)
        coarse = max(1, bins // 4)
        features = self.temporal(marked.log1p().reshape(batch * sensors, self.marks, bins))
        features = functional.adaptive_avg_pool1d(features, coarse).reshape(
            batch, sensors, -1, coarse
        )
        local_count = values.sum(-1).log1p()[..., None]
        total_count = values.sum((1, 2)).log1p()[:, None, None].expand(-1, sensors, -1)
        fraction = p.reshape(batch, 1, 1).expand(-1, sensors, -1)
        observed = torch.cat([coordinates, local_count, total_count, fraction], -1)
        sensor_features = self.sensor_context(torch.cat([features.mean(-1), observed], -1))
        context = (sensor_features * dom_live[..., None]).sum(1) / dom_live.sum(1, keepdim=True)
        context = context[:, None, :, None].expand(-1, sensors, -1, coarse)
        time = (
            torch.linspace(-1, 1, coarse, device=counts.device)
            .reshape(1, 1, 1, -1)
            .expand(batch, sensors, -1, -1)
        )
        decoder_input = torch.cat(
            [features, context, observed[..., None].expand(-1, -1, -1, coarse), time], 2
        )
        output = self.decoder(decoder_input.reshape(batch * sensors, -1, coarse))
        output = functional.interpolate(output, size=bins, mode="linear", align_corners=False)
        return output.reshape(batch, sensors * self.marks, bins)
