"""Full spatial/time convolutions on a fixed, instrumented detector lattice."""

import numpy as np
import torch
from torch import nn
from torch_convnd import Conv4d


class GridPhotonReference4d(nn.Module):
    """Predict sensor/time logits using only retained counts and detector metadata.

    Sensors occupy a subset of a regular Cartesian lattice. Uninstrumented cells
    have zero input features; output probabilities are normalized only after
    gathering the actual sensors. Sensor ordering does not define adjacency.
    """

    def __init__(self, positions_m, channels=4, backend="auto"):
        """Validate physical geometry and retain all input time bins."""
        super().__init__()
        positions = np.asarray(positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            message = "Expected finite sensor positions with shape [sensor,3]"
            raise ValueError(message)
        if not len(positions) or len(np.unique(positions, axis=0)) != len(positions):
            message = "Sensor positions must be nonempty and unique"
            raise ValueError(message)
        axes = [np.unique(np.round(positions[:, axis], 5)) for axis in range(3)]
        spacing = []
        for values in axes:
            differences = np.diff(values)
            step = float(np.mean(differences)) if len(differences) else 1.0
            if len(differences) and not np.allclose(differences, step, rtol=1e-4, atol=1e-4):
                message = "This dense-grid experiment requires regularly spaced coordinate axes"
                raise ValueError(message)
            spacing.append(step)
        self.grid_shape = tuple(len(axis) for axis in axes)
        if np.prod(self.grid_shape) > 4096:
            message = "Detector lattice exceeds the bounded 4096-cell pilot support"
            raise ValueError(message)
        self.register_buffer("origin_m", torch.tensor([axis[0] for axis in axes]).float())
        self.register_buffer("spacing_m", torch.tensor(spacing).float())
        self.register_buffer("grid_extent", torch.tensor(self.grid_shape, dtype=torch.long))
        self.stem = Conv4d(9, channels, 1, backend=backend)
        self.blocks = nn.ModuleList(
            Conv4d(
                channels,
                channels,
                3,
                padding=(1, 1, 1, dilation),
                dilation=(1, 1, 1, dilation),
                backend=backend,
            )
            for dilation in (1, 2, 4)
        )
        self.head = Conv4d(channels, 1, 1, backend=backend)
        self.activation = nn.SiLU()

    def sensor_indices(self, positions_m):
        """Map physical coordinates to cells, including permuted sensor orders."""
        coordinates = (positions_m.float() - self.origin_m) / self.spacing_m
        rounded = coordinates.round()
        if not torch.allclose(coordinates, rounded, atol=1e-3, rtol=0):
            message = "Sensor position is not on the declared detector lattice"
            raise ValueError(message)
        indices = rounded.long()
        if torch.any(indices < 0) or torch.any(indices >= self.grid_extent):
            message = "Sensor position lies outside the declared detector lattice"
            raise ValueError(message)
        flat = (indices[..., 0] * self.grid_shape[1] + indices[..., 1]) * self.grid_shape[2]
        flat = flat + indices[..., 2]
        if torch.any(flat.sort(dim=-1).values.diff(dim=-1) == 0):
            message = "Multiple sensors cannot share a detector lattice cell"
            raise ValueError(message)
        return flat

    def forward(self, counts, positions_m, live, p):
        """Return [batch,sensor,time] logits without target or truth inputs."""
        batch, sensors, bins = counts.shape
        if positions_m.ndim == 2:
            positions_m = positions_m[None].expand(batch, -1, -1)
        if live.ndim == 1:
            live = live[None].expand(batch, -1)
        if positions_m.shape != (batch, sensors, 3) or live.shape != (batch, sensors):
            message = "Counts, geometry and live support must describe the same sensors"
            raise ValueError(message)
        values = counts.float().masked_fill(~live[..., None], 0)
        index = self.sensor_indices(positions_m)
        coordinates = positions_m.float() / 500.0
        local = values.sum(-1).log1p()[..., None]
        total = values.sum((1, 2)).log1p()[:, None, None].expand(-1, sensors, -1)
        fraction = p.reshape(batch, 1, 1).expand(-1, sensors, -1)
        metadata = torch.cat([live[..., None], coordinates, local, total, fraction], -1)
        time = torch.linspace(-1, 1, bins, device=counts.device).expand(batch, sensors, -1)
        features = torch.cat(
            [
                values.log1p()[..., None],
                metadata[:, :, None].expand(-1, -1, bins, -1),
                time[..., None],
            ],
            -1,
        )
        features = features.masked_fill(~live[:, :, None, None], 0).permute(0, 3, 1, 2)
        cells = int(np.prod(self.grid_shape))
        grid = features.new_zeros(batch, 9, cells, bins).scatter(
            2, index[:, None, :, None].expand(-1, 9, -1, bins), features
        )
        hidden = self.activation(self.stem(grid.reshape(batch, 9, *self.grid_shape, bins)))
        for block in self.blocks:
            hidden = hidden + self.activation(block(hidden))
        logits = self.head(hidden).reshape(batch, cells, bins)
        return logits.gather(1, index[..., None].expand(-1, -1, bins))
