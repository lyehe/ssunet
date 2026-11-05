"""Run registered split-probability shifts and analytic detector-response diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from ..ssunet.datasets.neutrino import shard_sha256, split_photon_counts
from ..ssunet.models.neutrino_dense import DensePhotonReference
from ..ssunet.neutrino_evaluation import PhotonScoreRecords, photon_score_summary
from ..ssunet.neutrino_streaming import collect_training_reference, stream_scores
from .neutrino_analytic_study import _baseline_bank, open_analytic
from .neutrino_prepare import write_json
from .neutrino_study import (
    _assert_same_records,
    _atomic_scores,
    _bank_from_selection,
    _dataset,
    _score_bundle,
    _training_reference,
    verify_analysis_lock,
)

RESPONSES = (
    "unchanged",
    "background",
    "inactive",
    "timing",
    "efficiency",
    "merging",
    "dead_time",
    "afterpulse",
)


class ResponsePhotonDataset(Dataset):
    """Apply fixed response rules before thinning, never using the held-out split."""

    def __init__(self, base, response, settings):
        """Keep observation IDs and fixed geometry, with reproducible perturbations."""
        self.base, self.response, self.settings = base, response, settings
        for key in (
            "path",
            "split",
            "indices",
            "event_ids",
            "configuration_ids",
            "positions_m",
            "edges_ns",
            "seed",
            "p",
        ):
            setattr(self, key, getattr(base, key))
        self.training = False
        self.live = base.live.copy()
        generator = np.random.default_rng(settings["seed"])
        n_sensors = len(self.live)
        self.efficiency = generator.uniform(*settings["efficiency_range"], size=n_sensors)
        self.offset_bins = generator.choice([-1, 1], n_sensors) * round(
            settings["timing_offset_ns"] / np.diff(self.edges_ns)[0]
        )
        if response == "inactive":
            inactive = generator.permutation(n_sensors)[
                : round(n_sensors * settings["inactive_fraction"])
            ]
            self.live[inactive] = False
        if response not in RESPONSES:
            message = "Unknown registered response perturbation"
            raise ValueError(message)

    def __len__(self):
        """Return unchanged catalogue size, including empty observations."""
        return len(self.base)

    def counts(self, index):  # noqa: C901
        """Transform observations using a source-event-specific response RNG."""
        counts, dropped = self.base.counts(index)
        counts = counts.copy()
        event = int(self.event_ids[self.indices[index]])
        rng = np.random.default_rng(
            np.random.SeedSequence([self.settings["seed"], event, RESPONSES.index(self.response)])
        )
        settings, response = self.settings, self.response
        if response == "background":
            locations = rng.integers(
                counts.size, size=rng.poisson(settings["background_added_expected_photons"])
            )
            np.add.at(counts.ravel(), locations, 1)
        elif response == "inactive":
            counts[~self.live] = 0
        elif response == "efficiency":
            counts = rng.binomial(counts, self.efficiency[:, None]).astype(np.int64)
        elif response == "timing":
            sensors, bins = np.nonzero(counts)
            shifted = bins + self.offset_bins[sensors]
            keep = (shifted >= 0) & (shifted < counts.shape[1])
            result = np.zeros_like(counts)
            np.add.at(result, (sensors[keep], shifted[keep]), counts[sensors[keep], bins[keep]])
            dropped += int(counts[sensors[~keep], bins[~keep]].sum())
            counts = result
        elif response == "merging":
            counts = (counts > 0).astype(np.int64)
        elif response == "dead_time":
            result = np.zeros_like(counts)
            distance = int(np.ceil(settings["dead_time_ns"] / np.diff(self.edges_ns)[0]))
            for sensor in np.flatnonzero(counts.sum(1)):
                last = -distance
                for time_bin in np.flatnonzero(counts[sensor]):
                    if time_bin - last >= distance:
                        result[sensor, time_bin] = 1
                        last = int(time_bin)
            counts = result
        elif response == "afterpulse":
            extra = rng.binomial(counts, settings["afterpulse_probability"])
            delay = round(settings["afterpulse_delay_ns"] / np.diff(self.edges_ns)[0])
            if delay <= 0:
                message = "Afterpulse delay must span at least one recorded time bin"
                raise ValueError(message)
            counts[:, delay:] += extra[:, :-delay]
        return counts, dropped

    def __getitem__(self, index):
        """Split perturbed integer observations with the same frozen evaluation seed."""
        counts, dropped = self.counts(index)
        event = int(self.event_ids[self.indices[index]])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, event, 0]))
        a, b = split_photon_counts(counts, self.p, rng)
        return {
            "input": torch.from_numpy(a),
            "target": torch.from_numpy(b),
            "positions_m": torch.from_numpy(self.positions_m.astype(np.float32)),
            "live": torch.from_numpy(self.live),
            "p": torch.tensor(self.p, dtype=torch.float32),
            "event_id": torch.tensor(event),
            "outside_window": torch.tensor(dropped),
        }


def analytic_response_study(config, device):
    """Evaluate seed 42 under finite prescribed response shifts without refitting."""
    root = Path(config["data"]["output"])
    if not (root / "test_results.json").exists():
        message = "Finish the sealed analytic test before response diagnostics"
        raise ValueError(message)
    output = root / "response_diagnostics"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    lock = json.loads((root / "analysis_lock.json").read_text())
    selected = json.loads((root / "baseline_selection.json").read_text())
    checkpoint = lock["checkpoints"]["42"]
    if shard_sha256(checkpoint["path"]) != checkpoint["sha256"]:
        message = "The selected analytic checkpoint changed"
        raise ValueError(message)
    model = DensePhotonReference(config["model"]["channels"]).to(device)
    model.load_state_dict(
        torch.load(checkpoint["path"], map_location="cpu", weights_only=True)["state_dict"]
    )
    training, testing = open_analytic(config, "train"), open_analytic(config, "test")
    reference = collect_training_reference(training)
    reports = {}
    for response in RESPONSES:
        dataset = ResponsePhotonDataset(testing, response, config["stress"])
        bank = _baseline_bank(config, training, reference, device, selected)
        if response == "inactive":
            bank.live = torch.as_tensor(dataset.live)
            bank.support = bank.live[:, None].expand_as(bank.observed)
            bank._set_priors()
        started = time.perf_counter()
        records, values = stream_scores(
            dataset,
            batch_size=config["training"]["batch_size"],
            device=device,
            model=model,
            baselines=bank,
        )
        _atomic_scores(output / f"{response}.npz", records, values)
        reports[response] = _score_bundle(records, values, config, bank.winner)
        reports[response]["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / f"{response}.json", reports[response])
    write_json(
        output / "summary.json",
        {
            "scope": "prescribed_analytic_response_shifts; single_seed_42; exploratory",
            "checkpoint": checkpoint,
            "settings": config["stress"],
            "reports": reports,
            "intensity_changes": ["background", "inactive", "timing", "efficiency"],
            "independent_Poisson_assumption_violations": ["merging", "dead_time", "afterpulse"],
            "response_resolution_ns": float(np.diff(testing.edges_ns)[0]),
            "dead_time_contract": (
                "binned nonparalyzable response; at most one detection per accepted bin"
            ),
            "detector_calibration_claimed": False,
            "code_sha256": shard_sha256(Path(__file__)),
        },
    )
    training.close()
    testing.close()


def secondary_probability_views(config, device):
    """Evaluate the two registered p shifts after the primary sealed analysis."""
    root = Path(config["data"]["output"])
    if not (root / "test_results.json").exists():
        message = "The primary sealed test must finish before secondary views"
        raise ValueError(message)
    lock = verify_analysis_lock(config)
    output = root / "secondary_views"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    write_json(
        output / "analysis.json",
        {
            "primary_lock_sha256": shard_sha256(root / "analysis_lock.json"),
            "p_values": config["test"]["secondary_split_probabilities"],
            "scope": "predeclared_distribution_shift; no_refitting_or_primary_claim_revision",
            "code_sha256": shard_sha256(Path(__file__)),
        },
    )
    training, reference = _training_reference(config)
    bank = _bank_from_selection(config, training, reference, device)
    reports, joint_records, joint_values = {}, [], []
    for p in config["test"]["secondary_split_probabilities"]:
        testing = _dataset(config, "test", p=p)
        records, baseline = stream_scores(
            testing, batch_size=config["training"]["batch_size"], device=device, baselines=bank
        )
        per_seed = []
        for seed in config["training"]["seeds"]:
            model = DensePhotonReference(config["model"]["channels"]).to(device)
            checkpoint = torch.load(
                lock["checkpoints"][str(seed)]["path"], map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["state_dict"])
            actual, values = stream_scores(
                testing, batch_size=config["training"]["batch_size"], device=device, model=model
            )
            _assert_same_records(records, actual)
            _atomic_scores(output / f"p_{p}_seed_{seed}.npz", records, values)
            per_seed.append(values["model"])
        values = {**baseline, "model": np.mean(per_seed, axis=0)}
        _atomic_scores(output / f"p_{p}_mean_seed.npz", records, values)
        reports[str(p)] = _score_bundle(records, values, config, bank.winner)
        reports[str(p)]["per_seed_scores"] = [
            photon_score_summary(losses, records) for losses in per_seed
        ]
        joint_records.append(records)
        joint_values.append(values)
        testing.close()
    pooled = PhotonScoreRecords(
        *(
            np.concatenate([getattr(r, key) for r in joint_records])
            for key in ("event_ids", "configuration_ids", "retained_totals", "target_totals")
        )
    )
    pooled_values = {key: np.concatenate([v[key] for v in joint_values]) for key in joint_values[0]}
    joint = _score_bundle(pooled, pooled_values, config, bank.winner)
    write_json(
        output / "summary.json",
        {
            "views": reports,
            "pooled_secondary_views": joint,
            "group_bootstrap_keeps_views_together": True,
            "independent_source_groups": len(np.unique(pooled.configuration_ids)),
            "unique_source_events": len(np.unique(pooled.event_ids)),
            "view_records_are_not_extra_independent_events": True,
            "scope": "secondary_distribution_shift; primary_rule_and_result_unchanged",
        },
    )
    training.close()


def main():
    """Run a registered post-selection diagnostic stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("responses", "p-shifts"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.action == "responses":
        analytic_response_study(config, args.device)
    else:
        secondary_probability_views(config, args.device)


if __name__ == "__main__":
    main()
