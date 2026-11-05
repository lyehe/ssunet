"""Bounded-batch photon likelihoods, training priors and donor controls."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .neutrino_evaluation import PhotonScoreRecords
from .photon_reference import photon_count_log_probabilities


def require_frozen(dataset, partition=None):
    """Reject mutable evaluation draws or accidentally supplied test fitting data."""
    if dataset.training or (partition is not None and dataset.split != partition):
        message = "Frozen observations from the explicitly requested partition are required"
        raise ValueError(message)


def collect_training_reference(dataset):
    """Fit only training observations and retain scalar donor counts, not count maps."""
    require_frozen(dataset, "train")
    observed = np.zeros((len(dataset.live), len(dataset.edges_ns) - 1), dtype=np.float64)
    retained = np.zeros(len(dataset), dtype=np.int64)
    outside = empty = 0
    for index in range(len(dataset)):
        item = dataset[index]
        counts = item["input"].numpy() + item["target"].numpy()
        observed += counts
        retained[index] = int(item["input"].sum())
        outside += int(item["outside_window"])
        empty += not counts.any()
    return {
        "observed_counts": observed,
        "retained_totals": retained,
        "event_ids": dataset.event_ids[dataset.indices].copy(),
        "configuration_ids": dataset.configuration_ids[dataset.indices].copy(),
        "outside_window_photons": outside,
        "empty_observations": empty,
        "events": len(dataset),
        "photons_in_window": int(observed.sum()),
    }


def empty_score_records(dataset):
    """Allocate only O(events) scalar records in persisted catalogue order."""
    return PhotonScoreRecords(
        event_ids=dataset.event_ids[dataset.indices].copy(),
        configuration_ids=dataset.configuration_ids[dataset.indices].copy(),
        retained_totals=np.zeros(len(dataset), dtype=np.int64),
        target_totals=np.zeros(len(dataset), dtype=np.int64),
    )


def batch_records(records, batch, offset):
    """Validate row identity before associating scalar scores with source events."""
    number = len(batch["event_id"])
    selected = slice(offset, offset + number)
    if not np.array_equal(batch["event_id"].numpy(), records.event_ids[selected]):
        message = "Evaluation batch order differs from persisted event IDs"
        raise ValueError(message)
    records.retained_totals[selected] = batch["input"].sum((1, 2)).numpy()
    records.target_totals[selected] = batch["target"].sum((1, 2)).numpy()
    return selected


def _normal_kernel(coordinates, width, live=None):
    """Form a row-stochastic physical-space kernel, including exact identity."""
    if width == 0:
        weights = torch.eye(len(coordinates), dtype=torch.float64)
    elif width is None:
        weights = torch.ones((len(coordinates), len(coordinates)), dtype=torch.float64)
    else:
        distance = torch.cdist(coordinates.double(), coordinates.double())
        weights = torch.exp(-0.5 * (distance / width) ** 2)
    if live is not None:
        weights *= live[None, :]
    return weights / weights.sum(1, keepdim=True).clamp_min(1e-300)


class PhotonBaselineBank:
    """Hold O(sensor*time) priors/kernels and produce only current-batch PDFs."""

    def __init__(self, observed, positions, live, edges, settings, device="cpu"):
        """Use training counts and fixed geometry; targets are not constructor inputs."""
        self.device = torch.device(device)
        self.observed = torch.as_tensor(observed, dtype=torch.float64)
        self.live = torch.as_tensor(live, dtype=torch.bool)
        self.support = self.live[:, None].expand_as(self.observed)
        self.positions = torch.as_tensor(positions, dtype=torch.float64)
        self.times = torch.as_tensor((edges[:-1] + edges[1:]) / 2)[:, None]
        self.settings = settings
        self._time_kernels = {}
        self._space_kernels = {}
        self.selected_prior_parameters = {
            "time_ns": settings["frozen_prior_time_ns"],
            "pseudocount": settings["frozen_prior_pseudocount"],
        }
        self.selected_mixture_parameters = {
            "time_ns": 0,
            "space_m": 0,
            "prior_fraction": settings["frozen_mixture_prior_fraction"],
        }
        self.winner = "frozen_mixture"
        self.priors = {}
        self._set_priors()

    def _prior(self, time_ns, pseudocount):
        """Smooth training counts on the fixed acquisition clock."""
        key = ("prior_float64", time_ns)
        if key not in self._time_kernels:
            self._time_kernels[key] = _normal_kernel(self.times, time_ns)
        probability = self.observed @ self._time_kernels[key] + pseudocount * self.support
        probability *= self.support
        return probability / probability.sum()

    def _set_priors(self):
        """Keep only a few valid-support prior maps resident on the selected device."""
        settings = self.settings
        values = {
            "uniform": self.support.double() / self.support.sum(),
            "original_prior": self._prior(0, settings["original_prior_pseudocount"]),
            "frozen_prior": self._prior(
                settings["frozen_prior_time_ns"], settings["frozen_prior_pseudocount"]
            ),
            "selected_prior": self._prior(**self.selected_prior_parameters),
        }
        self.priors = {key: value.float().to(self.device) for key, value in values.items()}

    def smooth(self, inputs, time_ns, space_m):
        """Smooth observed A only, with physical kernels independent of B."""
        time_key, space_key = ("input_float32", str(self.device), time_ns), space_m
        if time_key not in self._time_kernels:
            self._time_kernels[time_key] = (
                _normal_kernel(self.times, time_ns).float().to(self.device)
            )
        if space_key not in self._space_kernels:
            self._space_kernels[space_key] = (
                _normal_kernel(self.positions, space_m, self.live).float().to(self.device)
            )
        temporal = inputs.float() @ self._time_kernels[time_key]
        return torch.einsum("is,bit->bst", self._space_kernels[space_key], temporal)

    @staticmethod
    def mixture(estimate, prior, fraction):
        """Mix normalized retained counts with a prior, retaining empty-input fallback."""
        total = estimate.sum((1, 2), keepdim=True)
        conditional = estimate / total.clamp_min(1)
        conditional = torch.where(total > 0, conditional, prior[None])
        return (1 - fraction) * conditional + fraction * prior

    def probabilities(self, inputs):
        """Construct a finite set of current-batch baseline distributions."""
        raw = inputs.float()
        selected = self.smooth(
            raw,
            self.selected_mixture_parameters["time_ns"],
            self.selected_mixture_parameters["space_m"],
        )
        return {
            "uniform": self.priors["uniform"].expand_as(raw),
            "original_prior": self.priors["original_prior"].expand_as(raw),
            "frozen_prior": self.priors["frozen_prior"].expand_as(raw),
            "fixed_mixture": self.mixture(raw, self.priors["original_prior"], 0.5),
            "frozen_mixture": self.mixture(
                raw, self.priors["frozen_prior"], self.settings["frozen_mixture_prior_fraction"]
            ),
            "selected_prior": self.priors["selected_prior"].expand_as(raw),
            "selected_mixture": self.mixture(
                selected,
                self.priors["selected_prior"],
                self.selected_mixture_parameters["prior_fraction"],
            ),
        }

    def state(self):
        """Serialize small training references and chosen settings without dense cohorts."""
        return {
            "observed_counts": self.observed.numpy(),
            "positions": self.positions.numpy(),
            "live": self.live.numpy(),
            "selected_prior": self.selected_prior_parameters,
            "selected_mixture": self.selected_mixture_parameters,
            "winner": self.winner,
        }


def _probability_losses(probability, target):
    """Compute finite raw categorical event losses without normalizing targets."""
    scored = probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
    return -(target * scored).sum((1, 2))


@torch.inference_mode()
def stream_scores(dataset, *, batch_size, device, model=None, baselines=None):
    """Reduce one dense batch at a time to scalar per-event score records."""
    require_frozen(dataset)
    records = empty_score_records(dataset)
    names = (["model"] if model is not None else []) + (
        list(
            baselines.probabilities(
                torch.zeros((1, len(dataset.live), len(dataset.edges_ns) - 1), device=device)
            )
        )
        if baselines is not None
        else []
    )
    values = {name: np.zeros(len(dataset), dtype=np.float64) for name in names}
    if model is not None:
        model.eval()
    positions = torch.as_tensor(dataset.positions_m, dtype=torch.float32, device=device)
    live = torch.as_tensor(dataset.live, device=device)
    offset = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        selected = batch_records(records, batch, offset)
        inputs, target = batch["input"].to(device), batch["target"].to(device)
        if model is not None:
            logits = model(inputs, positions, live, batch["p"].to(device))
            logq = photon_count_log_probabilities(logits, live).masked_fill(target == 0, 0)
            values["model"][selected] = (-(target * logq).sum((1, 2))).cpu().double().numpy()
        if baselines is not None:
            for name, probability in baselines.probabilities(inputs).items():
                values[name][selected] = (
                    _probability_losses(probability, target).cpu().double().numpy()
                )
        offset += len(inputs)
    if not all(np.isfinite(value).all() for value in values.values()):
        message = "Non-finite streaming photon likelihood"
        raise RuntimeError(message)
    return records, values


@torch.inference_mode()
def select_baselines(training_reference, validation, bank, batch_size):
    """Select the finite registered control grid on primary validation NLL only."""
    del training_reference
    require_frozen(validation, "validation")
    primary_target = torch.zeros_like(bank.observed)
    all_target = torch.zeros_like(bank.observed)
    for batch in DataLoader(validation, batch_size=batch_size):
        number = batch["input"].sum((1, 2))
        primary = (number >= 1) & (number <= 10)
        primary_target += batch["target"][primary].sum(0)
        all_target += batch["target"].sum(0)
    if not primary_target.sum():
        message = "Primary validation stratum contains no target photons"
        raise ValueError(message)
    candidates = []
    prior_best = np.inf
    for width in bank.settings["prior_time_ns"]:
        for pseudocount in bank.settings["prior_pseudocounts"]:
            probability = bank._prior(width, pseudocount)
            score = float(
                -(primary_target * probability.clamp_min(1e-300).log()).sum() / primary_target.sum()
            )
            candidates.append(
                {
                    "kind": "prior",
                    "time_ns": width,
                    "pseudocount": pseudocount,
                    "primary_nll": score,
                }
            )
            if score < prior_best:
                prior_best = score
                bank.selected_prior_parameters = {"time_ns": width, "pseudocount": pseudocount}
    bank._set_priors()
    parameters = [
        {"time_ns": width, "space_m": space, "prior_fraction": fraction}
        for width in bank.settings["input_time_ns"]
        for space in bank.settings["input_space_m"]
        for fraction in bank.settings["mixture_prior_fractions"]
    ]
    totals = torch.zeros((len(parameters), 2), dtype=torch.float64, device=bank.device)
    for batch in DataLoader(validation, batch_size=batch_size):
        inputs, target = batch["input"].to(bank.device), batch["target"].to(bank.device)
        number = inputs.sum((1, 2))
        primary = (number >= 1) & (number <= 10)
        cached, last_key = None, None
        for index, item in enumerate(parameters):
            key = (item["time_ns"], item["space_m"])
            if key != last_key:
                cached = bank.smooth(inputs, *key)
                last_key = key
            probability = bank.mixture(
                cached, bank.priors["selected_prior"], item["prior_fraction"]
            )
            losses = _probability_losses(probability, target).double()
            totals[index, 0] += losses[primary].sum()
            totals[index, 1] += losses.sum()
    losses = totals.cpu().numpy()
    primary_scores = losses[:, 0] / float(primary_target.sum())
    best = int(np.argmin(primary_scores))
    bank.selected_mixture_parameters = parameters[best]
    candidates.extend(
        {
            "kind": "mixture",
            **item,
            "primary_nll": float(primary_scores[index]),
            "all_nll": float(losses[index, 1] / float(all_target.sum())),
        }
        for index, item in enumerate(parameters)
    )
    records, scores = stream_scores(
        validation, batch_size=batch_size, device=bank.device, baselines=bank
    )
    mask = records.strata["retained_1_to_10"]
    primary_by_method = {
        name: float(value[mask].sum() / records.target_totals[mask].sum())
        for name, value in scores.items()
    }
    bank.winner = min(primary_by_method, key=primary_by_method.get)
    return {
        "candidates": candidates,
        "method_primary_nll": primary_by_method,
        "selected_method": bank.winner,
        "selected_prior": bank.selected_prior_parameters,
        "selected_mixture": bank.selected_mixture_parameters,
    }


class BoundedPhotonDonors:
    """Read count-matched training inputs through a bounded LRU cache."""

    def __init__(self, dataset, retained_totals, *, seed, cache_events):
        """Index only A counts; target values and validation labels are not consulted."""
        require_frozen(dataset, "train")
        self.dataset, self.seed, self.cache_events = dataset, seed, cache_events
        self.retained_totals = np.asarray(retained_totals)
        self.by_count = {
            int(number): np.flatnonzero(self.retained_totals == number)
            for number in np.unique(self.retained_totals)
        }
        self.counts = np.asarray(sorted(self.by_count))
        self.cache = OrderedDict()

    def choose(self, event_id, count, draw):
        """Choose reproducibly among exact or explicitly nearest count matches."""
        candidates = self.by_count.get(int(count))
        if candidates is None:
            distance = np.abs(self.counts - count)
            nearest = self.counts[distance == distance.min()]
            candidates = np.concatenate([self.by_count[int(number)] for number in nearest])
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, int(event_id), draw]))
        return int(generator.choice(candidates))

    def input_at(self, index):
        """Cache at most the declared number of current donor count maps."""
        if index not in self.cache:
            self.cache[index] = self.dataset[index]["input"]
            while len(self.cache) > self.cache_events:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


@torch.inference_mode()
def stream_donor_scores(model, validation, donors, *, batch_size, device, draws):
    """Score exact primary count matches while retaining original validation targets."""
    require_frozen(validation)
    if (
        validation.p != donors.dataset.p
        or not np.array_equal(validation.edges_ns, donors.dataset.edges_ns)
        or not np.array_equal(validation.live, donors.dataset.live)
        or not np.allclose(validation.positions_m, donors.dataset.positions_m)
    ):
        message = "Donor and target exposure, p or physical geometry differ"
        raise ValueError(message)
    model.eval()
    positions = torch.as_tensor(validation.positions_m, dtype=torch.float32, device=device)
    live = torch.as_tensor(validation.live, device=device)
    values = np.zeros(len(validation), dtype=float)
    exact = np.zeros(len(validation), dtype=bool)
    selected_ids = np.full((len(validation), draws), -1, dtype=np.int64)
    offset = 0
    for batch in DataLoader(validation, batch_size=batch_size):
        counts = batch["input"].sum((1, 2)).numpy()
        target = batch["target"].to(device)
        for draw in range(draws):
            indices = [
                donors.choose(event_id, number, draw)
                for event_id, number in zip(batch["event_id"].numpy(), counts, strict=True)
            ]
            selected_ids[offset : offset + len(counts), draw] = donors.dataset.event_ids[
                donors.dataset.indices[indices]
            ]
            exact[offset : offset + len(counts)] = donors.retained_totals[indices] == counts
            inputs = torch.stack([donors.input_at(index) for index in indices]).to(device)
            logits = model(inputs, positions, live, batch["p"].to(device))
            logq = photon_count_log_probabilities(logits, live).masked_fill(target == 0, 0)
            values[offset : offset + len(counts)] += (
                -(target * logq).sum((1, 2))
            ).cpu().double().numpy() / draws
        offset += len(counts)
    return values, exact, selected_ids
