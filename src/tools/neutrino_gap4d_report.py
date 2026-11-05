"""Replay saved GAP checkpoints and render both photon and event weighted scores."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from . import neutrino_gap4d_retry as study
from .neutrino_conv4d_visualize import render_results

if __package__ == "src.tools":
    from ..ssunet.neutrino_evaluation import PhotonScoreRecords
    from ..ssunet.photon_reference import photon_count_log_probabilities
else:
    from ssunet.neutrino_evaluation import PhotonScoreRecords
    from ssunet.photon_reference import photon_count_log_probabilities


def event_contrasts(saved, config, donor_coverage):
    """Reuse whole-configuration resampling with one unit per nonempty target."""
    totals = saved["target_totals"]
    records = PhotonScoreRecords(
        event_ids=saved["event_ids"],
        configuration_ids=saved["configuration_ids"],
        retained_totals=saved["retained_totals"],
        target_totals=(totals > 0).astype(np.int64),
    )
    candidate = saved["conv4d"].mean(0)
    comparisons = {}
    for name in (
        "dense_reference",
        "full_time_reference",
        "fixed_mixture",
        "previously_tuned_mixture",
        "conv4d_donor",
    ):
        values = saved[name]
        control = values.mean(0) if values.ndim == 2 else values
        delta = (candidate - control) / np.maximum(totals, 1)
        comparisons[name] = {}
        for stratum, mask in records.strata.items():
            if name == "conv4d_donor" and donor_coverage[stratum] != 1.0:
                comparisons[name][stratum] = None
                continue
            result = study.reference.paired_photon_interval(
                delta,
                records,
                mask & (totals > 0),
                seed=config["evaluation"]["bootstrap_seed"],
                draws=config["evaluation"]["bootstrap_draws"],
            )
            if result is not None:
                result["model_minus_control_mean_event_cross_entropy"] = result.pop(
                    "model_minus_control_nats_per_photon"
                )
            comparisons[name][stratum] = result
    return comparisons


def audit_and_render(root, device_name="cuda:0"):
    """Verify data/source identity, all final scores and physical probability support."""
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    process = json.loads((root / "process.json").read_text())
    if process["status"] != "complete":
        message = "Wait for the fixed training schedule to complete before auditing"
        raise ValueError(message)
    identity = json.loads((root / "identity.json").read_text())
    if study.identity(config) != identity:
        message = "Training data or source differs from its frozen identity"
        raise ValueError(message)
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    training, _, validation, _, _ = study.development_views(config)
    runs = []
    for settings in config["models"]:
        for seed in config["training"]["seeds"]:
            directory = root / settings["name"] / f"seed_{seed}"
            checkpoint = torch.load(directory / "last.pt", map_location=device, weights_only=True)
            if (
                checkpoint["state"]["step"] != config["training"]["steps"]
                or checkpoint["config"] != config
                or checkpoint["model"] != settings
                or checkpoint["seed"] != seed
            ):
                message = "Checkpoint does not match the completed fixed run"
                raise RuntimeError(message)
            model = study.make_model(settings, training.positions_m).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            replay = study.reference.model_event_nll(
                model, validation, device, config["training"]["batch_size"]
            )
            with np.load(directory / "scores.npz") as saved:
                difference = float(np.max(np.abs(replay - saved["model"])))
            if difference > 1e-5:
                message = "Checkpoint scores differ from the saved final evaluation"
                raise RuntimeError(message)
            with torch.inference_mode():
                logits = model(
                    validation.inputs[:16].to(device),
                    validation.positions_m.to(device),
                    validation.live.to(device),
                    torch.full((16,), validation.p, device=device),
                )
                probabilities = photon_count_log_probabilities(
                    logits, validation.live.to(device)
                ).exp()
                mass_error = float((probabilities.sum((1, 2)) - 1).abs().max())
                invalid_mass = float(probabilities[:, ~validation.live.to(device)].sum())
                finite = bool(torch.isfinite(probabilities).all())
            if not finite or mass_error > 2e-6 or invalid_mass != 0:
                message = "Predictions fail joint physical probability normalization"
                raise RuntimeError(message)
            runs.append(
                {
                    "model": settings["name"],
                    "seed": seed,
                    "steps": checkpoint["state"]["step"],
                    "checkpoint_sha256": study.reference.shard_sha256(directory / "last.pt"),
                    "maximum_event_nll_replay_error": difference,
                    "maximum_probability_mass_error": mass_error,
                    "invalid_support_mass": invalid_mass,
                }
            )
    report = json.loads((root / "results.json").read_text())
    with np.load(root / "per_event_scores.npz") as saved:
        report["event_mean_contrasts"] = event_contrasts(
            saved, config, report["donor_exact_match_fraction_by_stratum"]
        )
    study.reference.write_json(root / "results.json", report)
    render_results(root, report, config, validation, training.positions_m, device, study.make_model)
    audit = {
        "status": "passed",
        "dataset_and_all_frozen_sources_verified": True,
        "test_or_truth_used": False,
        "runs": runs,
        "report_source_sha256": study.reference.shard_sha256(__file__),
        "renderer_source_sha256": study.reference.shard_sha256(
            Path(__file__).with_name("neutrino_conv4d_visualize.py")
        ),
    }
    study.reference.write_json(root / "evidence_audit.json", audit)
    training.close()
    return audit


def main():
    """Audit completed training without fitting or selecting checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="data/neutrino/gap4d_retry_v1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    audit = audit_and_render(args.study, args.device)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger(__name__).info(
        "Audit %s; replayed %s checkpoints", audit["status"], len(audit["runs"])
    )


if __name__ == "__main__":
    main()
