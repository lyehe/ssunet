"""Fork the observed failing development seed before divergence for a matched FP32 test."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from ..ssunet.datasets.neutrino import shard_sha256
from ..ssunet.models.neutrino_precision import FullPrecisionPhotonReference
from ..ssunet.neutrino_streaming import stream_scores
from ..ssunet.neutrino_training import _save_torch, train_photon_seed
from .neutrino_ablations import mechanism_dataset
from .neutrino_prepare import write_json
from .neutrino_study import (
    _assert_same_records,
    _atomic_scores,
    _code_hashes,
    _config_hash,
    _dataset,
    _read_scalar_scores,
    _score_bundle,
    _training_reference,
)


def _configuration(settings):
    """Keep every optimizer/data option while declaring the short matched continuation."""
    main = yaml.safe_load(Path(settings["main_config"]).read_text())
    config = copy.deepcopy(main)
    config["training"]["seeds"] = [settings["seed"]]
    config["training"]["epochs"] = settings["end_epoch"]
    config["continuation"] = settings
    return config


def _identity(config):
    """Pin the parent weights and unchanged main development observations."""
    root = Path(config["data"]["output"])
    settings = config["continuation"]
    fresh = yaml.safe_load(Path(settings["fresh_test_config"]).read_text())
    parent = root / f"runs/seed_{settings['seed']}/epoch_{settings['prefix_epoch']:03d}.pt"
    code = _code_hashes()
    for path in (Path(__file__), Path(__file__).parents[1] / "ssunet/models/neutrino_precision.py"):
        code[str(path.resolve())] = shard_sha256(path)
    return {
        "config_sha256": _config_hash(config),
        "fresh_test_observations_sha256": shard_sha256(Path(fresh["data"]["output"]) / "test.h5"),
        "parent_checkpoint_sha256": shard_sha256(parent),
        "observations": {
            name: shard_sha256(root / f"{name}.h5") for name in ("train", "validation")
        },
        "code_sha256": code,
    }


def train_continuation(config, device):
    """Continue seed 43 with the identical batch cursor and FP32 forward/backward."""
    settings = config["continuation"]
    root, output = Path(config["data"]["output"]), Path(settings["output"])
    directory = output / f"runs/seed_{settings['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    identity = _identity(config)
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    parent_directory = root / f"runs/seed_{settings['seed']}"
    parent_path = parent_directory / f"epoch_{settings['prefix_epoch']:03d}.pt"
    if not (directory / "last.pt").exists():
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        if (
            parent["model_seed"] != settings["seed"]
            or parent["training_state"]["next_epoch"] != settings["prefix_epoch"]
        ):
            message = "The precision fork does not start at the declared parent cursor"
            raise ValueError(message)
        parent_identity = json.loads((root / "training_manifest.json").read_text())["identity"]
        if parent["identity"] != parent_identity:
            message = "The parent checkpoint differs from the original study identity"
            raise ValueError(message)
        # This is an explicit new experiment, not an altered original checkpoint.
        # Prefix weights, optimizer, RNG, cursor, counters and scores remain intact.
        parent["identity"], parent["config"] = identity, config
        _save_torch(directory / "last.pt", parent)
        _save_torch(directory / "best_low_count.pt", parent)
        shutil.copyfile(
            parent_directory / f"epoch_{settings['prefix_epoch']:03d}_scores.npz",
            directory / "best_validation_scores.npz",
        )
        write_json(
            output / "fork_provenance.json",
            {
                "parent_path": str(parent_path),
                "parent_sha256": identity["parent_checkpoint_sha256"],
                "original_parent_unmodified": True,
                "prefix_training_retained": "BF16 first two epochs; optimizer and RNG preserved",
                "only_continuation_change": "autocast disabled in the unchanged dense model",
                "configuration": config,
                "identity": identity,
            },
        )
    main = yaml.safe_load(Path(settings["main_config"]).read_text())
    training, _ = _training_reference(main)
    ids = training.event_ids[training.indices][
        np.random.default_rng(config["evaluation"]["diagnostic_seed"]).permutation(len(training))[
            : config["evaluation"]["fixed_training_diagnostic_events"]
        ]
    ]
    fitting, validation, diagnostic = (
        _dataset(config, "train", training=True),
        _dataset(config, "validation"),
        _dataset(config, "train", ids=ids),
    )
    result = train_photon_seed(
        config,
        settings["seed"],
        (fitting, validation, diagnostic),
        lambda: FullPrecisionPhotonReference(config["model"]["channels"]),
        directory,
        device,
        identity,
    )
    if result["completed_fixed_epochs"]:
        records, fp32 = _read_scalar_scores(
            directory / f"epoch_{settings['end_epoch']:03d}_scores.npz"
        )
        original, bf16 = _read_scalar_scores(
            parent_directory / f"epoch_{settings['end_epoch']:03d}_scores.npz"
        )
        _assert_same_records(records, original)
        report = _score_bundle(
            records,
            {"model": fp32["model"], "bf16_matched_epoch": bf16["model"]},
            config,
            "bf16_matched_epoch",
        )
        write_json(
            output / "validation_results.json",
            {
                "scope": (
                    "single_preidentified_development_failure; "
                    "matched_prefix_optimizer_data_and_updates"
                ),
                "evidence": report,
                "result": result,
                "causal_generalization_claimed": False,
            },
        )
        lock = {
            "identity": identity,
            "configuration": config,
            "models": {
                "fp32_epoch5": str(directory / f"epoch_{settings['end_epoch']:03d}.pt"),
                "bf16_epoch5": str(parent_directory / f"epoch_{settings['end_epoch']:03d}.pt"),
                "bf16_selected_epoch2": str(parent_path),
                "fp32_validation_selected": str(directory / "best_low_count.pt"),
            },
        }
        lock["model_sha256"] = {name: shard_sha256(path) for name, path in lock["models"].items()}
        if not (output / "analysis_lock.json").exists():
            write_json(output / "analysis_lock.json", lock)
    for dataset in (training, fitting, validation, diagnostic):
        dataset.close()


def test_continuation(config, device):
    """Score the four predeclared checkpoints on fresh mechanism families once."""
    output = Path(config["continuation"]["output"])
    if (output / "test_results.json").exists():
        message = "The matched precision-continuation test already has results"
        raise FileExistsError(message)
    lock = json.loads((output / "analysis_lock.json").read_text())
    if lock["identity"] != _identity(config):
        message = "The continuation implementation changed after selection"
        raise ValueError(message)
    fresh = yaml.safe_load(Path(config["continuation"]["fresh_test_config"]).read_text())
    testing = mechanism_dataset(fresh, "test")
    device = torch.device(device)
    torch.set_num_threads(config["training"]["cpu_threads"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    write_json(
        output / "test_opened.json",
        {
            "continuation_lock_sha256": shard_sha256(output / "analysis_lock.json"),
            "fresh_cohort_sha256": shard_sha256(
                Path(fresh["data"]["output"]) / "cohort_manifest.parquet"
            ),
        },
    )
    scores, reference_records = {}, None
    for name, path in lock["models"].items():
        if shard_sha256(path) != lock["model_sha256"][name]:
            message = "A registered continuation checkpoint changed"
            raise ValueError(message)
        model = FullPrecisionPhotonReference(config["model"]["channels"]).to(device)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        records, values = stream_scores(
            testing, batch_size=config["training"]["batch_size"], device=device, model=model
        )
        if reference_records is not None:
            _assert_same_records(reference_records, records)
        reference_records = records
        scores[name] = values["model"]
    _atomic_scores(output / "test_scores.npz", reference_records, scores)
    fixed = _score_bundle(
        reference_records,
        {"model": scores["fp32_epoch5"], "bf16_epoch5": scores["bf16_epoch5"]},
        config,
        "bf16_epoch5",
    )
    selected = _score_bundle(
        reference_records,
        {
            "model": scores["fp32_validation_selected"],
            "bf16_selected_epoch2": scores["bf16_selected_epoch2"],
        },
        config,
        "bf16_selected_epoch2",
    )
    write_json(
        output / "test_results.json",
        {
            "scope": "fresh_family_test_of_single_matched_precision_fork; exploratory",
            "fixed_epoch_contrast": fixed,
            "validation_selected_contrast": selected,
            "source_seed": config["continuation"]["seed"],
            "main_test_reused": False,
            "fresh_test_events": len(reference_records.event_ids),
        },
    )
    testing.close()


def main():
    """Run a single finite continuation, preserving the original failed experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "test"))
    parser.add_argument(
        "--config", type=Path, default=Path("config/neutrino_precision_continuation.yml")
    )
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = _configuration(yaml.safe_load(args.config.read_text()))
    if args.action == "train":
        train_continuation(config, args.device)
    else:
        test_continuation(config, args.device)


if __name__ == "__main__":
    main()
