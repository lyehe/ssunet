import torch

EPSILON = 1e-8


def mse_loss(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    result = result * mask if mask is not None else result
    target = target * mask if mask is not None else target
    exp_energy = torch.exp(result)
    exp_energy = (
        exp_energy / torch.mean(exp_energy, dim=(-1, -2, -3, -4), keepdim=True)
        + EPSILON
    )
    target = target / (
        (torch.mean(target, dim=(-1, -2, -3, -4), keepdim=True)) + EPSILON
    )
    return torch.mean((exp_energy - target) ** 2)


def l1_loss(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    result = result * mask if mask is not None else result
    target = target * mask if mask is not None else target
    exp_energy = torch.exp(result)
    exp_energy = (
        exp_energy / torch.mean(exp_energy, dim=(-1, -2, -3, -4), keepdim=True)
        + EPSILON
    )
    target = target / (
        (torch.mean(target, dim=(-1, -2, -3, -4), keepdim=True)) + EPSILON
    )
    return torch.mean(torch.abs(exp_energy - target))


def photon_loss(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    result = result * mask if mask is not None else result
    target = target * mask if mask is not None else target
    exp_energy = torch.exp(result)
    per_image = -torch.mean(result * target, dim=(-1, -2, -3, -4), keepdim=True)
    per_image += torch.log(
        torch.mean(exp_energy, dim=(-1, -2, -3, -4), keepdim=True) + EPSILON
    ) * torch.mean(target, dim=(-1, -2, -3, -4), keepdim=True)
    return torch.mean(per_image)


def photon_loss_2D(
    result: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    result = result * mask if mask is not None else result
    target = target * mask if mask is not None else target
    exp_energy = torch.exp(result)
    per_image = -torch.mean(result * target, dim=(-1, -2, -4), keepdim=True)
    per_image += torch.log(
        torch.mean(exp_energy, dim=(-1, -2, -4), keepdim=True) + EPSILON
    ) * torch.mean(target, dim=(-1, -2, -4), keepdim=True)
    return torch.mean(per_image)
