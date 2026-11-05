"""Preserve full projection precision in a controlled unchanged-backbone variant."""

import torch

from .neutrino_dense import DensePhotonReference


class FullPrecisionPhotonReference(DensePhotonReference):
    """Use the identical architecture and weights with autocast disabled."""

    def forward(self, counts, positions_m, live, p):
        """Keep relative logits and their backward projection in FP32 under outer AMP."""
        with torch.autocast(device_type=counts.device.type, enabled=False):
            return super().forward(counts.float(), positions_m.float(), live, p.float())
