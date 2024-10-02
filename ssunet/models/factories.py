"""Factories for the models."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

import pyiqa
import torch

from ssunet.loss import loss_functions


class Metric(ABC):
    """Abstract class for metrics."""

    @abstractmethod
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the metric."""


# Define the type alias for the loss function
LossFunction = Callable[..., torch.Tensor]


def create_loss_function(name: str) -> LossFunction:
    """Create a loss function."""
    return loss_functions[name]


def create_psnr_metric(device: torch.device) -> Metric:
    """Create a PSNR metric."""
    return cast(Metric, pyiqa.create_metric("psnr", device=device))


def create_ssim_metric(device: torch.device) -> Metric:
    """Create a SSIM metric."""
    return cast(Metric, pyiqa.create_metric("ssim", channels=1, device=device))
