"""Marked-pulse controls with O(DOM*time*marks) storage and no DOM-pair kernel."""

import torch
import torch.nn.functional as functional


def smooth_pulse_time(values, width):
    """Apply a mass-preserving finite temporal kernel on the fixed source clock."""
    if width == 0:
        return values
    radius = int(3 * width)
    coordinates = torch.arange(-radius, radius + 1, dtype=values.dtype, device=values.device)
    kernel = torch.exp(-0.5 * (coordinates / width) ** 2)
    kernel /= kernel.sum()
    kernel = kernel.reshape(1, 1, -1)
    bins = values.shape[-1]
    edge_mass = functional.conv1d(
        torch.ones((1, 1, bins), dtype=values.dtype, device=values.device), kernel, padding=radius
    )
    scaled = values / edge_mass.reshape(*([1] * (values.ndim - 1)), bins)
    return functional.conv1d(scaled.reshape(-1, 1, bins), kernel, padding=radius).reshape_as(values)


class MarkedPulseBaselines:
    """Build current-batch marked distributions from training observations and A."""

    def __init__(self, observed, settings, device="cpu", *, selected=None):
        """Preserve the training-only prior and a finite validation-selected control."""
        prior = torch.as_tensor(observed, dtype=torch.float64, device=device)
        prior = smooth_pulse_time(prior, 1) + settings["prior_pseudocount"]
        prior /= prior.sum()
        self.prior = prior.float()
        self.width = 1 if selected is None else selected["time_width_bins"]
        self.fraction = 0.25 if selected is None else selected["prior_fraction"]
        self.winner = "selected_pulse_mixture"

    def mixture(self, inputs, width, fraction):
        """Smooth whole retained pulses with their marks; never use held-out charge."""
        observed = smooth_pulse_time(inputs.float(), width)
        total = observed.sum((1, 2), keepdim=True)
        empirical = torch.where(total > 0, observed / total.clamp_min(1), self.prior[None])
        return (1 - fraction) * empirical + fraction * self.prior

    def probabilities(self, inputs):
        """Return only a small number of current-batch control maps."""
        return {
            "uniform_marked": torch.full_like(inputs, 1 / self.prior.numel(), dtype=torch.float32),
            "training_marked_prior": self.prior[None].expand_as(inputs),
            self.winner: self.mixture(inputs, self.width, self.fraction),
        }
