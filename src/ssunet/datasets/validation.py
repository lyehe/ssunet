"""Reproducible held-out photon splits with optional image references."""

import torch
import torchvision.transforms.v2.functional as tf

from ..configs import DataConfig, SplitParams, SSUnetData
from ..constants import EPSILON
from ..utils import _normalize_by_mean
from .singlevolume import SingleVolumeDataset


class ValidationDataset(SingleVolumeDataset):
    """Validate photon prediction on a fixed split of each sampled observation.

    The first two items are the held-out target and remaining input photons.
    A supplied secondary image is an optional third item for image diagnostics;
    it is never used to create the photon target. Sampling uses a local generator
    indexed by sample, so epochs and DataLoader workers see the same validation
    pairs without resetting the training random-number generators.
    """

    def __init__(
        self,
        input: SSUnetData,
        config: DataConfig,
        split_params: SplitParams | None = None,
        seed: int = 0,
        **kwargs,
    ) -> None:
        """Configure fixed splitting using the training split policy when supplied."""
        super().__init__(input, config, **kwargs)
        self.split_params = split_params
        self.seed = seed if split_params is None or split_params.seed is None else split_params.seed

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        """Return a reproducible photon split and any separately supplied reference."""
        if index < 0 or index >= len(self):
            message = "Validation sample index is out of range"
            raise IndexError(message)
        generator = torch.Generator(device=self.data.device)
        generator.manual_seed((int(self.seed) + index) % (2**63))

        if self.config.virtual_size:
            start = int(
                torch.randint(self.data_size, (), generator=generator, device=self.data.device)
            )
        else:
            start = index * self.config.skip_frames
        observation = self.data[start : start + self.z_size]
        items = [observation]
        if self.secondary_data is not None:
            items.append(self.secondary_data[start : start + self.z_size])
        items = self._transform(items, generator)

        counts = items[0].float().clamp_min(0).floor()
        p = self._sample_p(counts, generator)
        noise = torch.binomial(counts, torch.full_like(counts, p), generator=generator)
        target = counts - noise
        normalize = (
            self.config.normalize_target
            if self.split_params is None
            else self.split_params.normalize_target
        )
        if normalize:
            target = _normalize_by_mean(target)
        output = [target, noise]
        if len(items) == 2:
            output.append(items[1].float())
        return self._add_channel_dim(output)

    def _transform(
        self, items: list[torch.Tensor], generator: torch.Generator
    ) -> list[torch.Tensor]:
        """Apply crop and augmentation using only the sample's local generator."""
        device = items[0].device
        if self.config.rotation > 0:
            angle = float(torch.rand((), generator=generator, device=device)) * self.config.rotation
            items = [tf.rotate(item, angle) for item in items]
        h_max = items[0].shape[-2] - self.x_size
        w_max = items[0].shape[-1] - self.y_size
        if h_max < 0 or w_max < 0:
            message = "Validation crop must fit within the observation"
            raise ValueError(message)
        if self.config.random_crop:
            h0 = int(torch.randint(h_max + 1, (), generator=generator, device=device))
            w0 = int(torch.randint(w_max + 1, (), generator=generator, device=device))
        else:
            h0, w0 = h_max // 2, w_max // 2
        self.crop_idx = (h0, w0, h0 + self.x_size, w0 + self.y_size)
        items = [item[..., h0 : h0 + self.x_size, w0 : w0 + self.y_size] for item in items]
        if self.config.augments:
            if float(torch.rand((), generator=generator, device=device)) < 0.5:
                items = [item.transpose(-1, -2) for item in items]
            if float(torch.rand((), generator=generator, device=device)) < 0.5:
                items = [item.flip(-1) for item in items]
            if float(torch.rand((), generator=generator, device=device)) < 0.5:
                items = [item.flip(-2) for item in items]
        return items

    def _sample_p(self, counts: torch.Tensor, generator: torch.Generator) -> float:
        """Draw the same per-index splitting probability on every validation pass."""
        if self.split_params is None:
            return 0.5
        params = self.split_params
        uniform = float(torch.rand((), generator=generator, device=counts.device))
        if params.method == "fixed":
            p = params.p_list[0] if params.p_list else params.min_p
        elif params.method == "list" and params.p_list:
            choice = min(int(uniform * len(params.p_list)), len(params.p_list) - 1)
            p = params.p_list[choice]
        else:
            p = params.min_p + uniform * (params.max_p - params.min_p)
            if params.method == "db":
                p = (10 ** (p / 10.0)) / (float(counts.mean()) + EPSILON)
                p = max(EPSILON, min(1 - EPSILON, p))
        if not 0 < p < 1:
            message = "Validation photon splitting probability must be between zero and one"
            raise ValueError(message)
        return float(p)

    @property
    def data_size(self) -> int:
        """Get the number of complete temporal windows."""
        return self.data.shape[0] - self.z_size + 1
