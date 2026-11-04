"""Run the finite matched-budget architecture and supervision comparison."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from ..ssunet.datasets.neutrino import NeutrinoPhotonDataset, shard_sha256
from ..ssunet.models.neutrino_dense import DensePhotonReference
from ..ssunet.models.neutrino_variants import FinePhotonReference, PriorResidualPhotonReference
from ..ssunet.neutrino_streaming import (
    _normal_kernel,
    collect_training_reference,
    select_baselines,
    stream_scores,
)
from ..ssunet.neutrino_training import train_photon_seed
from .neutrino_analytic_study import _baseline_bank
from .neutrino_prepare import write_json
from .neutrino_study import (
    _assert_same_records,
    _atomic_scores,
    _code_hashes,
    _config_hash,
    _score_bundle,
)


def mechanism_dataset(config, partition, *, training=False, ids=None):
    """Read the fresh family-disjoint mechanism cohort without physical labels."""
    return NeutrinoPhotonDataset(
        Path(config["data"]["output"]) / f"{partition}.h5",
        partition,
        lazy=True,
        training=training,
        seed=config["data"]["photon_split_seed"],
        p=config["data"]["split_probability"],
        selected_event_ids=ids,
    )


def mechanism_model(config, architecture, reference, training):
    """Create exactly one registered model with no access to B or latent truth."""
    if architecture == "unchanged":
        return DensePhotonReference(config["model"]["channels"])
    if architecture == "full_temporal_resolution":
        return FinePhotonReference(config["model"]["channels"])
    if architecture != "prior_mixture_residual":
        message = "Unknown prospective mechanism model"
        raise ValueError(message)
    settings = config["residual"]
    observed = torch.as_tensor(reference["observed_counts"], dtype=torch.float64)
    times = torch.as_tensor((training.edges_ns[:-1] + training.edges_ns[1:]) / 2)[:, None]
    prior = (
        observed @ _normal_kernel(times, settings["prior_time_ns"]) + settings["prior_pseudocount"]
    )
    prior /= prior.sum()
    kernel = _normal_kernel(times, settings["input_time_ns"])
    return PriorResidualPhotonReference(
        prior, kernel, fraction=settings["prior_fraction"], channels=config["model"]["channels"]
    )


def mechanism_identity(config):
    """Bind all variants to the same observations, protocol and code revision."""
    root = Path(config["data"]["output"])
    code = _code_hashes()
    for path in (Path(__file__), Path(__file__).parents[1] / "ssunet/models/neutrino_variants.py"):
        code[str(path.resolve())] = shard_sha256(path)
    return {
        "config_sha256": _config_hash(config),
        "code_sha256": code,
        "shards_sha256": {
            name: shard_sha256(root / f"{name}.h5") for name in ("train", "validation", "test")
        },
    }


def train_mechanisms(config, device):
    """Train four hypotheses with identical groups, epochs, ordering and seed budgets."""
    main = yaml.safe_load(Path(config["main_study_config"]).read_text())
    main_root = Path(main["data"]["output"])
    first_seed = main_root / "runs/seed_42/training_result.json"
    if not (main_root / "development_summary.json").exists() and not first_seed.exists():
        message = "A completed unchanged development seed is required before mechanism runs"
        raise ValueError(message)
    root = Path(config["data"]["output"])
    device = torch.device(device)
    torch.set_num_threads(config["training"]["cpu_threads"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    training, validation = (
        mechanism_dataset(config, "train"),
        mechanism_dataset(config, "validation"),
    )
    reference = collect_training_reference(training)
    identity = mechanism_identity(config)
    manifest = root / "training_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
        message = "Mechanism study implementation changed after its evidence freeze"
        raise ValueError(message)
    write_json(
        manifest,
        {
            "identity": identity,
            "conditional_rationale": (
                "all-event resolution/estimator gap in existing pilot; "
                "fresh-vs-fixed supervision mechanism"
            ),
            "main_test_outcomes_used_for_design": False,
            "parallel_timing": "separate_GPU_after_first_unchanged_seed; main_protocol_untouched",
        },
    )
    baseline_config = {**config, "baselines": main["baselines"]}
    if (root / "baseline_selection.json").exists():
        selected = json.loads((root / "baseline_selection.json").read_text())
        bank = _baseline_bank(baseline_config, training, reference, device, selected)
    else:
        bank = _baseline_bank(baseline_config, training, reference, device)
        selected = select_baselines(reference, validation, bank, config["training"]["batch_size"])
        write_json(root / "baseline_selection.json", selected)
    diagnostic_ids = training.event_ids[training.indices][
        : config["evaluation"]["fixed_training_diagnostic_events"]
    ]
    diagnostic = mechanism_dataset(config, "train", ids=diagnostic_ids)
    runs = {}
    for variant in config["variants"]:
        fitting = mechanism_dataset(config, "train", training=variant["fresh_splits"])
        runs[variant["name"]] = []
        for seed in config["training"]["seeds"]:
            result = train_photon_seed(
                config,
                seed,
                (fitting, validation, diagnostic),
                lambda variant=variant: mechanism_model(
                    config, variant["architecture"], reference, training
                ),
                root / f"runs/{variant['name']}/seed_{seed}",
                device,
                {**identity, "variant": variant},
            )
            runs[variant["name"]].append(result)
            if not result["completed_fixed_epochs"]:
                write_json(root / "development_progress.json", {"runs": runs})
                return
        fitting.close()
    checkpoints = {
        variant["name"]: {
            str(seed): {
                "path": str(root / f"runs/{variant['name']}/seed_{seed}/best_low_count.pt"),
                "sha256": shard_sha256(
                    root / f"runs/{variant['name']}/seed_{seed}/best_low_count.pt"
                ),
            }
            for seed in config["training"]["seeds"]
        }
        for variant in config["variants"]
    }
    lock = {
        "identity": identity,
        "checkpoints": checkpoints,
        "runs": runs,
        "baseline_selection_sha256": shard_sha256(root / "baseline_selection.json"),
    }
    if not (root / "analysis_lock.json").exists():
        write_json(root / "analysis_lock.json", lock)
    for dataset in (training, validation, diagnostic):
        dataset.close()


def test_mechanisms(config, device):
    """Evaluate frozen variants on their fresh independent test exactly once."""
    root = Path(config["data"]["output"])
    if (root / "test_results.json").exists():
        message = "The mechanism test already has results"
        raise FileExistsError(message)
    lock = json.loads((root / "analysis_lock.json").read_text())
    if lock["identity"] != mechanism_identity(config):
        message = "Mechanism analysis differs from its selected implementation"
        raise ValueError(message)
    if lock["baseline_selection_sha256"] != shard_sha256(root / "baseline_selection.json"):
        message = "Mechanism baseline changed after selection"
        raise ValueError(message)
    device = torch.device(device)
    torch.set_num_threads(config["training"]["cpu_threads"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    write_json(
        root / "test_opened.json",
        {
            "analysis_lock_sha256": shard_sha256(root / "analysis_lock.json"),
            "claim": "exploratory_mechanism; fresh_main_and_pilot_family_disjoint_test",
        },
    )
    training, testing = mechanism_dataset(config, "train"), mechanism_dataset(config, "test")
    reference = collect_training_reference(training)
    main = yaml.safe_load(Path(config["main_study_config"]).read_text())
    selected = json.loads((root / "baseline_selection.json").read_text())
    bank = _baseline_bank(
        {**config, "baselines": main["baselines"]}, training, reference, device, selected
    )
    records, values = stream_scores(
        testing, batch_size=config["training"]["batch_size"], device=device, baselines=bank
    )
    for variant in config["variants"]:
        losses = []
        for seed in config["training"]["seeds"]:
            checkpoint = lock["checkpoints"][variant["name"]][str(seed)]
            if shard_sha256(checkpoint["path"]) != checkpoint["sha256"]:
                message = "A selected mechanism checkpoint changed"
                raise ValueError(message)
            model = mechanism_model(config, variant["architecture"], reference, training).to(device)
            model.load_state_dict(
                torch.load(checkpoint["path"], map_location="cpu", weights_only=True)["state_dict"]
            )
            actual, scored = stream_scores(
                testing, batch_size=config["training"]["batch_size"], device=device, model=model
            )
            _assert_same_records(records, actual)
            _atomic_scores(root / f"test_{variant['name']}_seed_{seed}.npz", records, scored)
            losses.append(scored["model"])
        values[variant["name"]] = np.mean(losses, axis=0)
    _atomic_scores(root / "test_mean_seed_scores.npz", records, values)
    comparisons = {
        variant["name"]: _score_bundle(
            records,
            {
                "model": values[variant["name"]],
                "dense_fresh": values["dense_fresh"],
                bank.winner: values[bank.winner],
            },
            config,
            "dense_fresh",
        )
        for variant in config["variants"]
    }
    write_json(
        root / "test_results.json",
        {
            "scope": (
                "finite_exploratory_mechanism_comparison; "
                "matched_training_examples_updates_and_caps"
            ),
            "comparisons": comparisons,
            "baseline": bank.winner,
            "independent_test_events": len(records.event_ids),
            "independent_source_groups": len(np.unique(records.configuration_ids)),
            "seed_count_is_not_extra_events": True,
            "original_main_test_not_reused": True,
        },
    )
    training.close()
    testing.close()


def main():
    """Execute one frozen mechanism study stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "test"))
    parser.add_argument("--config", type=Path, default=Path("config/neutrino_ablations.yml"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = yaml.safe_load(args.config.read_text())
    if args.action == "train":
        train_mechanisms(config, args.device)
    else:
        test_mechanisms(config, args.device)


if __name__ == "__main__":
    main()
