"""An explicitly empirical marked-pulse adapter with no PE-to-photon conversion."""

import numpy as np
import torch

from .neutrino import NeutrinoPhotonDataset


class MarkedPulseDataset(NeutrinoPhotonDataset):
    """Read one row per reconstructed pulse in the shared ragged storage layout."""

    def __init__(self, *args, **kwargs):
        """Require the pulse contract before using the shared count tensor machinery."""
        super().__init__(*args, **kwargs)
        if (
            self.provenance.get("kind") != "empirical_real_marked_pulses"
            or self.provenance.get("unit_photon_detections") is not False
            or self.provenance.get("charge_is_photon_count") is not False
        ):
            message = "MarkedPulseDataset requires the explicit non-photon pulse contract"
            raise ValueError(message)
        self.charge_categories = len(self.provenance["charge_interior_edges_pe"]) + 1

    def __getitem__(self, index):
        """Thin whole pulses, including their charge marks; never split PE amplitude."""
        index, epoch = index if isinstance(index, tuple) else (index, self.epoch)
        source = self.indices[index]
        counts, dropped = self.counts(index)
        event = int(self.event_ids[source])
        epoch = epoch if self.training else 0
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, event, epoch]))
        retained = np.zeros_like(counts)
        # Sampling only nonzero multiplicities avoids millions of zero-bin RNG
        # operations for the full detector. The binomial law is unchanged.
        nonzero = np.flatnonzero(counts)
        retained.ravel()[nonzero] = generator.binomial(counts.ravel()[nonzero], self.p)
        return {
            "input": torch.from_numpy(retained),
            "target": torch.from_numpy(counts - retained),
            "positions_m": torch.from_numpy(self.positions_m.astype(np.float32)),
            "live": torch.from_numpy(self.live),
            "p": torch.tensor(self.p, dtype=torch.float32),
            "event_id": torch.tensor(event),
            "outside_window": torch.tensor(dropped),
        }
