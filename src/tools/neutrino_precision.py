"""Test the development-triggered FP32 fix with the identical dense backbone."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from ..ssunet.datasets.neutrino import shard_sha256
from ..ssunet.models.neutrino_precision import FullPrecisionPhotonReference
from ..ssunet.neutrino_streaming import stream_scores
from ..ssunet.neutrino_training import train_photon_seed
from .neutrino_ablations import mechanism_dataset
from .neutrino_prepare import write_json
from .neutrino_study import (
    _assert_same_records,
    _atomic_scores,
    _code_hashes,
    _config_hash,
    _read_scalar_scores,
    _score_bundle,
)


def resolve_config(settings):
    """Inherit the matched-budget protocol while recording the single precision change."""
    config = yaml.safe_load(Path(settings["mechanism_config"]).read_text())
    return {
        **config,
        "precision_followup": settings,
        "training": {**config["training"], "precision_override": "full_float32_inside_forward"},
    }


def precision_identity(config):
    """Pin the shared untouched mechanism observations and the additional hypothesis."""
    root = Path(config["data"]["output"])
    code = _code_hashes()
    for path in (Path(__file__), Path(__file__).parents[1] / "ssunet/models/neutrino_precision.py"):
        code[str(path.resolve())] = shard_sha256(path)
    return {
        "configuration_sha256": _config_hash(config),
        "code_sha256": code,
        "observations": {
            name: shard_sha256(root / f"{name}.h5") for name in ("train", "validation", "test")
        },
        "mechanism_baseline_sha256": shard_sha256(root / "baseline_selection.json"),
    }


def train_precision(config, device):
    """Train three seeds with equal examples/updates and unchanged optimizer settings."""
    output = Path(config["precision_followup"]["output"])
    output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    identity = precision_identity(config)
    manifest = output / "training_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
        message = "The precision follow-up changed after its evidence freeze"
        raise ValueError(message)
    write_json(
        manifest,
        {
            "identity": identity,
            "configuration": config,
            "hypothesis": (
                "FP32 may reduce observed common-logit drift; not assumed causal until compared"
            ),
            "main_test_used_for_design": False,
        },
    )
    training, validation = (
        mechanism_dataset(config, "train"),
        mechanism_dataset(config, "validation"),
    )
    diagnostic_ids = training.event_ids[training.indices][
        : config["evaluation"]["fixed_training_diagnostic_events"]
    ]
    diagnostic = mechanism_dataset(config, "train", ids=diagnostic_ids)
    fitting = mechanism_dataset(config, "train", training=True)
    results = []
    for seed in config["training"]["seeds"]:
        result = train_photon_seed(
            config,
            seed,
            (fitting, validation, diagnostic),
            lambda: FullPrecisionPhotonReference(config["model"]["channels"]),
            output / f"runs/seed_{seed}",
            device,
            identity,
        )
        results.append(result)
        if not result["completed_fixed_epochs"]:
            write_json(output / "development_progress.json", {"runs": results})
            return
    lock = {
        "identity": identity,
        "checkpoints": {
            str(seed): {
                "path": str(output / f"runs/seed_{seed}/best_low_count.pt"),
                "sha256": shard_sha256(output / f"runs/seed_{seed}/best_low_count.pt"),
            }
            for seed in config["training"]["seeds"]
        },
        "runs": results,
        "test_outcomes_used_for_design": False,
    }
    if not (output / "analysis_lock.json").exists():
        write_json(output / "analysis_lock.json", lock)
    for dataset in (training, validation, diagnostic, fitting):
        dataset.close()


def test_precision(config, device):
    """Compare the sealed precision hypothesis with frozen BF16 reference test scores."""
    output = Path(config["precision_followup"]["output"])
    root = Path(config["data"]["output"])
    if (output / "test_results.json").exists():
        message = "The precision follow-up test already has results"
        raise FileExistsError(message)
    if not (root / "test_results.json").exists():
        message = "Complete the sealed original mechanism comparison before the added contrast"
        raise ValueError(message)
    lock = json.loads((output / "analysis_lock.json").read_text())
    if lock["identity"] != precision_identity(config):
        message = "The precision hypothesis changed after model selection"
        raise ValueError(message)
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    write_json(
        output / "test_opened.json",
        {
            "precision_lock_sha256": shard_sha256(output / "analysis_lock.json"),
            "original_mechanism_lock_sha256": shard_sha256(root / "analysis_lock.json"),
        },
    )
    testing = mechanism_dataset(config, "test")
    original_records, original_values = _read_scalar_scores(root / "test_mean_seed_scores.npz")
    losses = []
    for seed in config["training"]["seeds"]:
        item = lock["checkpoints"][str(seed)]
        if shard_sha256(item["path"]) != item["sha256"]:
            message = "The selected FP32 checkpoint changed"
            raise ValueError(message)
        model = FullPrecisionPhotonReference(config["model"]["channels"]).to(device)
        model.load_state_dict(
            torch.load(item["path"], map_location="cpu", weights_only=True)["state_dict"]
        )
        records, values = stream_scores(
            testing, batch_size=config["training"]["batch_size"], device=device, model=model
        )
        _assert_same_records(records, original_records)
        _atomic_scores(output / f"test_seed_{seed}.npz", records, values)
        losses.append(values["model"])
    baseline = json.loads((root / "baseline_selection.json").read_text())["selected_method"]
    values = {
        "model": np.mean(losses, axis=0),
        "dense_fresh": original_values["dense_fresh"],
        baseline: original_values[baseline],
    }
    _atomic_scores(output / "test_mean_seed_scores.npz", original_records, values)
    write_json(
        output / "test_results.json",
        {
            "scope": (
                "development_triggered_precision_mechanism; "
                "matched_observations_optimizer_epochs_and_caps"
            ),
            "versus_mixed_precision": _score_bundle(
                original_records, values, config, "dense_fresh"
            ),
            "versus_simple_baseline": _score_bundle(original_records, values, config, baseline),
            "independent_events": len(original_records.event_ids),
            "independent_source_groups": len(np.unique(original_records.configuration_ids)),
            "original_main_test_not_reused": True,
            "architecture_and_parameter_count_unchanged": True,
        },
    )
    testing.close()


def main():
    """Run a fixed precision comparison without changing the ongoing main study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "test"))
    parser.add_argument("--config", type=Path, default=Path("config/neutrino_precision.yml"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = resolve_config(yaml.safe_load(args.config.read_text()))
    if args.action == "train":
        train_precision(config, args.device)
    else:
        test_precision(config, args.device)


if __name__ == "__main__":
    main()
