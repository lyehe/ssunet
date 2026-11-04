"""Adapt the released GAP residual U-Net to a detector sensor/time matrix."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from ..vendor.gap.GAP_UNET_ResBlock import UN


class PublishedGAPReference(nn.Module):
    """Preserve the published network and explicitly map detector observations.

    Sensor rows follow lexicographic x/y/z order: each vertical detector string
    is contiguous. Geometry determines row order, without becoming an extra
    network feature. Padding is used only inside the backbone and removed
    before probability normalization. No target counts enter this adapter.
    """

    def __init__(self, positions_m, *, depth=6, start_filts=28, levels=10):
        """Use the architecture configured in the GAP paper and released example."""
        super().__init__()
        positions = np.asarray(positions_m, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            message = "Expected finite physical positions with shape [sensor,3]"
            raise ValueError(message)
        if len(np.unique(positions, axis=0)) != len(positions):
            message = "Detector positions must be distinct"
            raise ValueError(message)
        order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
        self.register_buffer("ordered_positions_m", torch.as_tensor(positions[order]))
        self.pooling_multiple = 2 ** (depth - 1)
        self.network = UN(
            channels=1,
            levels=levels,
            depth=depth,
            start_filts=start_filts,
            up_mode="transpose",
            merge_mode="add",
        )

    def _sensor_mapping(self, positions_m, batch):
        """Keep the same physical image when caller sensor indices are permuted."""
        if positions_m.ndim == 2:
            positions_m = positions_m[None].expand(batch, -1, -1)
        matches = torch.isclose(
            positions_m[:, :, None].float(),
            self.ordered_positions_m[None, None],
            rtol=0,
            atol=1e-4,
        ).all(-1)
        if not torch.all(matches.sum(1) == 1) or not torch.all(matches.sum(2) == 1):
            message = "Observations must contain each configured detector position once"
            raise ValueError(message)
        return matches.to(torch.int64).argmax(1), matches.to(torch.int64).argmax(2)

    def forward(self, counts, positions_m, live, p):
        """Return logits on the original sensors/time bins from raw A counts alone."""
        batch, sensors, bins = counts.shape
        if live.ndim == 1:
            live = live[None].expand(batch, -1)
        to_image, from_image = self._sensor_mapping(positions_m, batch)
        values = counts.float().masked_fill(~live[..., None], 0)
        image = values.gather(1, to_image[..., None].expand(-1, -1, bins))[:, None]
        pad_sensors = (-sensors) % self.pooling_multiple
        pad_time = (-bins) % self.pooling_multiple
        image = functional.pad(image, (0, pad_time, 0, pad_sensors))
        logits = self.network(image)[:, 0, :sensors, :bins]
        return logits.gather(1, from_image[..., None].expand(-1, -1, bins))
