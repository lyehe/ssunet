"""Frozen observed-photon views and likelihood controls for the neutrino pilot."""

from dataclasses import dataclass

import numpy as np
import torch

from .datasets.neutrino import NeutrinoPhotonDataset
from .photon_reference import photon_count_log_probabilities


def retained_count_strata(number) -> dict[str, np.ndarray]:
    """Define evaluation strata from retained observations, independently of targets."""
    return {
        "all": np.ones(len(number), dtype=bool),
        "retained_0": number == 0,
        "retained_1_to_5": (number >= 1) & (number <= 5),
        "retained_6_to_10": (number >= 6) & (number <= 10),
        "retained_1_to_10": (number >= 1) & (number <= 10),
        "retained_11_to_30": (number >= 11) & (number <= 30),
        "retained_above_30": number > 30,
    }


@dataclass
class PhotonScoreRecords:
    """Store only scalar event evidence instead of cohort-sized dense maps."""

    event_ids: np.ndarray
    configuration_ids: np.ndarray
    retained_totals: np.ndarray
    target_totals: np.ndarray

    @property
    def strata(self):
        """Return the same input-count strata used by the frozen pilot."""
        return retained_count_strata(self.retained_totals)


@dataclass
class FrozenPhotonView:
    """Keep targets for scoring, separate from the explicitly forwarded inputs."""

    inputs: torch.Tensor
    targets: torch.Tensor
    positions_m: torch.Tensor
    live: torch.Tensor
    p: float
    event_ids: np.ndarray
    configuration_ids: np.ndarray

    @property
    def target_totals(self) -> np.ndarray:
        """Return observed target photon totals without event normalization."""
        return self.targets.sum(dim=(1, 2)).numpy()

    @property
    def strata(self) -> dict[str, np.ndarray]:
        """Select diagnostic strata using retained inputs only."""
        number = self.inputs.sum(dim=(1, 2)).numpy()
        return retained_count_strata(number)


def freeze_photon_view(dataset: NeutrinoPhotonDataset, indices=None) -> FrozenPhotonView:
    """Cache observed counts with an explicit frozen split, without truth access."""
    if dataset.training:
        message = "Evaluation views require a frozen photon-split dataset"
        raise ValueError(message)
    selected = np.arange(len(dataset)) if indices is None else np.asarray(indices)
    if not len(selected):
        message = "Cannot evaluate an empty event partition"
        raise ValueError(message)
    items = [dataset[int(index)] for index in selected]
    source = dataset.indices[selected]
    return FrozenPhotonView(
        inputs=torch.stack([item["input"] for item in items]),
        targets=torch.stack([item["target"] for item in items]),
        positions_m=torch.as_tensor(dataset.positions_m, dtype=torch.float32),
        live=torch.as_tensor(dataset.live),
        p=dataset.p,
        event_ids=dataset.event_ids[source],
        configuration_ids=dataset.configuration_ids[source],
    )


def probability_nll(probability: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Score per-event count likelihoods, including zeros on invalid support."""
    if target.is_floating_point() or target.dtype == torch.bool or (target < 0).any():
        message = "Likelihood scoring requires nonnegative integer photon targets"
        raise ValueError(message)
    if probability.shape != target.shape or not torch.isfinite(probability).all():
        message = "Probability and target arrays must be finite and have matching shapes"
        raise ValueError(message)
    mass = probability.sum(dim=(1, 2))
    if (probability < 0).any() or not torch.allclose(mass, torch.ones_like(mass), atol=1e-6):
        message = "Each event requires a normalized nonnegative probability distribution"
        raise ValueError(message)
    if ((target > 0) & (probability == 0)).any():
        message = "A target photon lies outside the predicted support"
        raise ValueError(message)
    logq = probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
    return -(target * logq).sum(dim=(1, 2)).cpu().numpy().astype(np.float64)


def fit_pilot_baselines(training, view: FrozenPhotonView, settings: dict) -> dict:
    """Fit priors on training observations with previously fixed hyperparameters."""
    observed = torch.zeros_like(view.inputs[0], dtype=torch.float64)
    for index in range(len(training)):
        observed += torch.from_numpy(training.counts(index)[0])
    support = view.live[:, None].expand_as(observed)
    original = observed + settings["original_prior_pseudocount"] * support
    original /= original.sum()
    times = torch.as_tensor((training.edges_ns[:-1] + training.edges_ns[1:]) / 2)
    bandwidth = settings["tuned_prior_time_bandwidth_ns"]
    distance = times[:, None] - times[None, :]
    kernel = torch.exp(-0.5 * (distance / bandwidth) ** 2)
    kernel /= kernel.sum(dim=1, keepdim=True)
    tuned = observed @ kernel + settings["tuned_prior_pseudocount"] * support
    tuned /= tuned.sum()
    retained = view.inputs.double() / view.inputs.sum(dim=(1, 2), keepdim=True).clamp_min(1)
    empty = view.inputs.sum(dim=(1, 2)) == 0
    fixed_retained, tuned_retained = retained.clone(), retained.clone()
    fixed_retained[empty] = original
    tuned_retained[empty] = tuned
    fraction = settings["tuned_mixture_prior_fraction"]
    distributions = {
        "uniform": (support.double() / support.sum()).expand_as(view.inputs),
        "original_training_prior": original.expand_as(view.inputs),
        "fixed_mixture": 0.5 * fixed_retained + 0.5 * original,
        "previously_tuned_training_prior": tuned.expand_as(view.inputs),
        "previously_tuned_mixture": (1 - fraction) * tuned_retained + fraction * tuned,
    }
    return {
        name: probability_nll(probability, view.targets)
        for name, probability in distributions.items()
    }


@torch.inference_mode()
def model_event_nll(model, view: FrozenPhotonView, device, batch_size, inputs=None) -> np.ndarray:
    """Forward observations and geometry only, then score separately with targets."""
    retained = view.inputs if inputs is None else inputs
    if retained.shape != view.inputs.shape:
        message = "Control input shape disagrees with the frozen event geometry"
        raise ValueError(message)
    model.eval()
    live, positions = view.live.to(device), view.positions_m.to(device)
    values = []
    for begin in range(0, len(retained), batch_size):
        counts = retained[begin : begin + batch_size].to(device)
        logits = model(counts, positions, live, torch.full((len(counts),), view.p, device=device))
        logq = photon_count_log_probabilities(logits, live)
        target = view.targets[begin : begin + batch_size].to(device)
        scored = logq.masked_fill(target == 0, 0)
        values.extend((-(target * scored).sum(dim=(1, 2))).cpu().double().tolist())
    result = np.asarray(values)
    if not np.isfinite(result).all():
        message = "Non-finite frozen photon likelihood"
        raise RuntimeError(message)
    return result


def matched_donor_indices(donor_counts, retained_counts, *, seed, draws) -> np.ndarray:
    """Select training donors by nearest retained photon total, without targets."""
    candidates = []
    for number in retained_counts:
        distance = np.abs(donor_counts - number)
        candidates.append(np.flatnonzero(distance == distance.min()))
    generator = np.random.default_rng(seed)
    return np.asarray(
        [[generator.choice(candidate) for candidate in candidates] for _ in range(draws)]
    )


def photon_score_summary(losses: np.ndarray, view: FrozenPhotonView) -> dict:
    """Report event counts and photon-weighted scores in input-defined strata."""
    totals = view.target_totals
    return {
        name: {
            "events": int(mask.sum()),
            "events_with_target": int(np.count_nonzero(totals[mask])),
            "held_out_photons": int(totals[mask].sum()),
            "nats_per_photon": float(losses[mask].sum() / totals[mask].sum())
            if totals[mask].sum()
            else None,
        }
        for name, mask in view.strata.items()
    }


def paired_photon_interval(
    delta, view: FrozenPhotonView | PhotonScoreRecords, mask, *, seed, draws, chunk_size=64
) -> dict | None:
    """Bootstrap whole persisted configurations for a photon-weighted difference."""
    groups, totals = view.configuration_ids, view.target_totals
    unique, inverse = np.unique(groups[mask], return_inverse=True)
    if not len(unique) or not totals[mask].sum():
        return None
    if draws < 1 or chunk_size < 1:
        message = "Bootstrap draw and chunk counts must be positive"
        raise ValueError(message)
    group_delta = np.bincount(inverse, weights=np.asarray(delta)[mask])
    group_total = np.bincount(inverse, weights=totals[mask])
    generator = np.random.default_rng(seed)
    chunks = []
    for first in range(0, draws, chunk_size):
        samples = generator.integers(
            len(unique), size=(min(chunk_size, draws - first), len(unique))
        )
        denominator = group_total[samples].sum(axis=1)
        numerator = group_delta[samples].sum(axis=1)
        valid = denominator > 0
        chunks.append(numerator[valid] / denominator[valid])
    values = np.concatenate(chunks)
    return {
        "model_minus_control_nats_per_photon": float(delta[mask].sum() / totals[mask].sum()),
        "descriptive_95_percent_interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "valid_bootstrap_draws": len(values),
        "resampling_unit": "persisted_configuration_id",
        "includes_model_seed_or_selection_uncertainty": False,
    }
