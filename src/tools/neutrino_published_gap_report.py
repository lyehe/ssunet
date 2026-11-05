"""Replay the published GAP comparison and export distribution-based evidence."""

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch

from . import neutrino_published_gap as study

if __package__ == "src.tools":
    from ..ssunet.neutrino_evaluation import PhotonScoreRecords
    from ..ssunet.photon_reference import photon_count_log_probabilities
else:
    from ssunet.neutrino_evaluation import PhotonScoreRecords
    from ssunet.photon_reference import photon_count_log_probabilities


def event_intervals(config, view, saved):
    """Use the same configuration bootstrap with one vote per nonempty target event."""
    totals = view.target_totals
    records = PhotonScoreRecords(
        event_ids=view.event_ids,
        configuration_ids=view.configuration_ids,
        retained_totals=view.inputs.sum((1, 2)).numpy(),
        target_totals=(totals > 0).astype(np.int64),
    )
    candidate = saved["conv4d"].mean(0)
    results = {}
    for settings in config["models"]:
        name = settings["name"]
        difference = (candidate - saved[name].mean(0)) / np.maximum(totals, 1)
        results[name] = {}
        for stratum, mask in records.strata.items():
            interval = study.retry.reference.paired_photon_interval(
                difference,
                records,
                mask & (totals > 0),
                seed=config["evaluation"]["bootstrap_seed"],
                draws=config["evaluation"]["bootstrap_draws"],
            )
            if interval is not None:
                interval["model_minus_control_mean_event_cross_entropy"] = interval.pop(
                    "model_minus_control_nats_per_photon"
                )
            results[name][stratum] = interval
    return results


def _audit_seed(root, config, settings, seed, positions, validation, device):
    """Verify one final published checkpoint and its exact saved observation scores."""
    directory = root / settings["name"] / f"seed_{seed}"
    saved = torch.load(directory / "last.pt", map_location=device, weights_only=True)
    expected = copy.deepcopy(config)
    expected["training"].update(settings.get("training_overrides", {}))
    expected["training"]["gradient_clip"] = settings["gradient_clip"]
    if (
        saved["config"] != expected
        or saved["model"] != settings
        or saved["seed"] != seed
        or saved["state"]["step"] != expected["training"]["steps"]
    ):
        message = "Published checkpoint does not match the final fixed protocol"
        raise ValueError(message)
    model = study.make_model(settings, positions).to(device)
    model.load_state_dict(saved["state_dict"])
    replay = study.retry.reference.model_event_nll(
        model, validation, device, expected["training"]["batch_size"]
    )
    with np.load(directory / "scores.npz") as scores:
        difference = float(np.abs(replay - scores["model"]).max())
    with torch.inference_mode():
        logits = model(
            validation.inputs[:16].to(device),
            validation.positions_m.to(device),
            validation.live.to(device),
            torch.full((16,), validation.p, device=device),
        )
        q = photon_count_log_probabilities(logits, validation.live.to(device)).exp()
        mass_error = float((q.sum((1, 2)) - 1).abs().max())
        invalid_mass = float(q[:, ~validation.live.to(device)].sum())
        finite = bool(torch.isfinite(q).all())
    if difference > 1e-5 or mass_error > 2e-6 or invalid_mass != 0 or not finite:
        message = "Published checkpoint failed score replay or physical PMF normalization"
        raise RuntimeError(message)
    return {
        "model": settings["name"],
        "seed": seed,
        "steps": saved["state"]["step"],
        "maximum_event_nll_replay_error": difference,
        "maximum_probability_mass_error": mass_error,
        "invalid_support_mass": invalid_mass,
        "checkpoint_sha256": study.retry.reference.shard_sha256(directory / "last.pt"),
    }


def audit_and_render(root, device_name="cuda:0"):
    """Bind the published code and reference study, then replay all final fits."""
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    process = json.loads((root / "process.json").read_text())
    if process["status"] != "complete":
        message = "All predeclared published fits must complete before final auditing"
        raise ValueError(message)
    if study.study_identity(config) != json.loads((root / "identity.json").read_text()):
        message = "Published comparison data, source or reference study changed"
        raise ValueError(message)
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    training, _, validation, _, _ = study.retry.development_views(config)
    study._validate_reference(config, validation)
    runs = [
        _audit_seed(root, config, settings, seed, training.positions_m, validation, device)
        for settings in config["models"]
        for seed in config["training"]["seeds"]
    ]
    report = json.loads((root / "results.json").read_text())
    with np.load(root / "per_event_scores.npz") as saved:
        report["event_mean_contrasts"] = event_intervals(config, validation, saved)
    study.write_json(root / "results.json", report)
    snapshot = json.loads((root / "source_snapshot.json").read_text())
    for record in snapshot["files"]:
        if study.retry.reference.shard_sha256(root / record["snapshot"]) != record["sha256"]:
            message = "An archived comparison source differs from its frozen bytes"
            raise ValueError(message)
    from .neutrino_published_gap_visualize import render_comparison

    render_comparison(root, report, config, validation, training.positions_m, device)
    audit = {
        "status": "passed",
        "test_or_truth_used": False,
        "data_training_published_and_reference_source_identities_verified": True,
        "source_snapshot_files_verified": len(snapshot["files"]),
        "published_network_source_unmodified": True,
        "published_upstream_commit": config["upstream_commit"],
        "published_upstream_source_sha256": config["upstream_source_sha256"],
        "runs": runs,
        "reference_checkpoint_audit": str(Path(config["reference_study"]) / "evidence_audit.json"),
        "report_source_sha256": study.retry.reference.shard_sha256(__file__),
        "renderer_source_sha256": study.retry.reference.shard_sha256(
            Path(__file__).with_name("neutrino_published_gap_visualize.py")
        ),
    }
    study.write_json(root / "evidence_audit.json", audit)
    training.close()
    return audit


def main():
    """Replay the complete published-network comparison without new training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="data/neutrino/published_gap_comparison_v1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audit = audit_and_render(args.study, args.device)
    logging.getLogger(__name__).info(
        "Audit %s: %s published fits", audit["status"], len(audit["runs"])
    )


if __name__ == "__main__":
    main()
