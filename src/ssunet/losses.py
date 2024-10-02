"""Loss functions."""

import torch

from .constants import EPSILON


def mse_loss(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the mean squared error loss.

    This function calculates the mean squared error (MSE) loss between the predicted result
    and the target. The MSE is defined as the average of the squared differences between
    the predicted and actual values.

    The function applies the following steps:
    1. If a mask is provided, both result and target are multiplied by the mask.
    2. The result is exponentiated (exp_energy = exp(result)).
    3. Both exp_energy and target are normalized by their respective means.
    4. The MSE is computed as the mean of (exp_energy - target)^2.

    Note: The normalization step may affect the interpretation of the MSE as a standard
    error metric. This implementation appears to be a custom variation of MSE.
    """
    result = result * mask if mask is not None else result
    target = target * mask if mask is not None else target
    exp_energy = torch.exp(result)
    exp_energy = exp_energy / (torch.mean(exp_energy, dim=(-1, -2, -3, -4), keepdim=True) + EPSILON)
    target = target / (torch.mean(target, dim=(-1, -2, -3, -4), keepdim=True) + EPSILON)
    return torch.mean((exp_energy - target) ** 2)


def l1_loss(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the mean absolute error (L1) loss.

    This function calculates the L1 loss between the predicted result and the target.
    The L1 loss is defined as the average of the absolute differences between
    the predicted and actual values.

    The function applies the following steps:
    1. If a mask is provided, both result and target are multiplied by the mask.
    2. The result is exponentiated (exp_energy = exp(result)).
    3. Both exp_energy and target are normalized by their respective means.
    4. The L1 loss is computed as the mean of |exp_energy - target|.
    """
    result = result * mask if mask is not None else result
    target = target * mask if mask is not None else target
    exp_energy = torch.exp(result)
    exp_energy = exp_energy / (torch.mean(exp_energy, dim=(-1, -2, -3, -4), keepdim=True) + EPSILON)
    target = target / (torch.mean(target, dim=(-1, -2, -3, -4), keepdim=True) + EPSILON)
    return torch.mean(torch.abs(exp_energy - target))


def photon_loss(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the photon loss.

    This function calculates a loss similar to cross-entropy, but adapted for photon-count data.
    It's designed to measure the discrepancy between the predicted photon distribution (result)
    and the actual photon distribution (target).

    The loss is computed as follows:
    1. If a mask is provided, both result and target are multiplied by the mask.
    2. The result is exponentiated (exp_energy = exp(result)).
    3. The negative mean of element-wise multiplication of result and target is calculated.
    4. The log of the mean of exp_energy is added, weighted by the mean of the target.

    This formulation is similar to cross-entropy in that it involves log-probabilities (result)
    and true probabilities (target), but it's adapted to handle the specific characteristics
    of photon distributions.
    """
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
    """Compute the photon loss for 2D data.

    Similar to the 3D version, but for use with 2D pairs instead of 3D volumes.
    """
    result = result * mask if mask is not None else result
    target = target * mask if mask is not None else target
    exp_energy = torch.exp(result)
    per_image = -torch.mean(result * target, dim=(-1, -2, -4), keepdim=True)
    per_image += torch.log(
        torch.mean(exp_energy, dim=(-1, -2, -4), keepdim=True) + EPSILON
    ) * torch.mean(target, dim=(-1, -2, -4), keepdim=True)
    return torch.mean(per_image)


loss_functions = {
    "mse": mse_loss,
    "l1": l1_loss,
    "photon": photon_loss,
    "photon_2d": photon_loss_2d,
}
