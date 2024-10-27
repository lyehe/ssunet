"""Bernoulli dataset."""

import torch

from .binomial import BinomDataset


class BernoulliDataset(BinomDataset):
    """Special case of the BinomDataset where the noise is sampled with a Bernoulli distribution."""

    @staticmethod
    def _sample_noise(input: torch.Tensor, p_value: float) -> torch.Tensor:
        return torch.bernoulli(torch.clamp(input * p_value, 0, 1))
