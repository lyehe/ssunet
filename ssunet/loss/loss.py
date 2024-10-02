"""Loss functions."""

from abc import ABC, abstractmethod

import torch

from ssunet.constants import EPSILON


class LossFunction(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def __call__(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss."""


def mse_loss(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the mean squared error loss."""
    result = result * mask if mask is not None else result
    target = target * mask if mask is not None else target
    exp_energy = torch.exp(result)
    exp_energy = exp_energy / torch.mean(exp_energy, dim=(-1, -2, -3, -4), keepdim=True) + EPSILON
    target = target / ((torch.mean(target, dim=(-1, -2, -3, -4), keepdim=True)) + EPSILON)
    return torch.mean((exp_energy - target) ** 2)


def l1_loss(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the mean absolute error loss."""
    result = result * mask if mask is not None else result
    target = target * mask if mask is not None else target
    exp_energy = torch.exp(result)
    exp_energy = exp_energy / torch.mean(exp_energy, dim=(-1, -2, -3, -4), keepdim=True) + EPSILON
    target = target / ((torch.mean(target, dim=(-1, -2, -3, -4), keepdim=True)) + EPSILON)
    return torch.mean(torch.abs(exp_energy - target))


def photon_loss(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the photon loss."""
    result = result * mask if mask is not None else result
    target = target * mask if mask is not None else target
    exp_energy = torch.exp(result)
    per_image = -torch.mean(result * target, dim=(-1, -2, -3, -4), keepdim=True)
    per_image += torch.log(
        torch.mean(exp_energy, dim=(-1, -2, -3, -4), keepdim=True) + EPSILON
    ) * torch.mean(target, dim=(-1, -2, -3, -4), keepdim=True)
    return torch.mean(per_image)


def photon_loss_2d(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the photon loss for 2D data."""
    result = result * mask if mask is not None else result
    target = target * mask if mask is not None else target
    exp_energy = torch.exp(result)
    per_image = -torch.mean(result * target, dim=(-1, -2, -4), keepdim=True)
    per_image += torch.log(
        torch.mean(exp_energy, dim=(-1, -2, -4), keepdim=True) + EPSILON
    ) * torch.mean(target, dim=(-1, -2, -4), keepdim=True)
    return torch.mean(per_image)
