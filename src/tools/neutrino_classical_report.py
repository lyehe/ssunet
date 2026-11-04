"""Audit locked classical photon estimators and render their neural comparison."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from . import neutrino_classical as study


def _check_partitions(root, training, validation):
    """Verify configuration separation and that the selection lock preceded validation access."""
    manifest = json.loads((root / "inner_split.json").read_text())
    access = json.loads((root / "validation_access.json").read_text())
    locked = study.shard_sha256(root / "selection_locked.json")
    fit = set(manifest["fit_event_ids"])
    tune = set(manifest["tuning_event_ids"])
    if (
        fit & tune
        or set(manifest["fit_configuration_ids"]) & set(manifest["tuning_configuration_ids"])
        or fit | tune != set(training.event_ids.tolist())
        or (fit | tune) & set(validation.event_ids.tolist())
        or not access["selection_locked_before_validation_open"]
        or access["selection_sha256"] != locked
        or manifest["outer_validation_photons_opened"]
    ):
        message = "Classical comparison failed the training-only selection contract"
        raise ValueError(message)
    return locked


def _check_predictions(root, validation, prior, probabilities, losses):
    """Verify every saved density, event NLL and physical probability support."""
    records = []
    with (
        np.load(root / "predictions.npz") as saved,
        np.load(root / "per_event_scores.npz") as scores,
    ):
        if not np.array_equal(saved["inputs"], validation.inputs.numpy()) or not np.array_equal(
            saved["held_out"], validation.targets.numpy()
        ):
            message = "Classical saved observations disagree with the fixed A/B split"
            raise ValueError(message)
        if np.max(np.abs(saved["prior"] - prior.cpu().numpy())) > 1e-12:
            message = "Refitted training-only population prior differs from the saved prior"
            raise ValueError(message)
        for name, q in probabilities.items():
            difference = float(np.abs(q - saved[name]).max())
            nll_error = float(np.abs(losses[name] - scores[name]).max())
            mass_error = float(np.abs(q.sum((1, 2)) - 1).max())
            invalid_mass = float(q[:, ~validation.live.numpy()].sum())
            if (
                difference > 1e-10
                or nll_error > 1e-8
                or mass_error > 1e-10
                or invalid_mass != 0
                or not np.isfinite(q).all()
                or (q < 0).any()
            ):
                message = "Classical prediction failed replay or probability-support validation"
                raise RuntimeError(message)
            records.append(
                {
                    "estimator": name,
                    "maximum_probability_replay_error": difference,
                    "maximum_event_nll_replay_error": nll_error,
                    "maximum_probability_mass_error": mass_error,
                    "invalid_support_mass": invalid_mass,
                }
            )
    return records


def audit_and_render(root, device_name="cuda:0"):
    """Recompute the prior and all deterministic predictions without new parameter selection."""
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    process = json.loads((root / "process.json").read_text())
    if process["status"] != "complete":
        message = "Wait for locked classical evaluation to finish before auditing"
        raise ValueError(message)
    if study.study_identity(config) != json.loads((root / "identity.json").read_text()):
        message = "Classical fitting source, data or reference artifacts changed"
        raise ValueError(message)
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    options = {"p": config["split_probability"], "seed": config["photon_split_seed"], "lazy": True}
    dataset = study.NeutrinoPhotonDataset(config["dataset"], "train", training=False, **options)
    training = study.freeze_photon_view(dataset)
    times = torch.as_tensor((dataset.edges_ns[:-1] + dataset.edges_ns[1:]) / 2).to(device)
    dataset.close()
    data = study.NeutrinoPhotonDataset(config["dataset"], "validation", **options)
    validation = study.freeze_photon_view(data)
    data.close()
    locked = _check_partitions(root, training, validation)
    selection = json.loads((root / "selection_locked.json").read_text())
    report = json.loads((root / "results.json").read_text())
    if locked != report["selection_sha256"] or report["selection"] != selection:
        message = "Reported classical settings differ from the locked training selection"
        raise ValueError(message)
    positions, live = training.positions_m.to(device), training.live.to(device)
    with torch.inference_mode():
        prior = study.fit_population_prior(
            (training.inputs + training.targets).to(device),
            positions,
            times,
            live,
            **selection["prior"],
            pseudocount=config["prior_pseudocount"],
        )
        probabilities, losses, numerical = study.predict_classical(
            config, validation, positions, times, prior, selection["estimators"], device
        )
    records = _check_predictions(root, validation, prior, probabilities, losses)
    report_replayed, _ = study.summarize(config, validation, losses, selection)
    for name in ("scores", "event_mean_nll", "conv4d_minus_classical", "event_mean_contrasts"):
        if report_replayed[name] != report[name]:
            message = "Classical aggregate scores or paired intervals do not replay"
            raise ValueError(message)
    snapshot = json.loads((root / "source_snapshot.json").read_text())
    for row in snapshot["files"]:
        if study.shard_sha256(root / row["snapshot"]) != row["sha256"]:
            message = "An archived classical source no longer matches its identity"
            raise ValueError(message)
    from .neutrino_classical_visualize import render_comparison

    render_comparison(root, report, config, validation, probabilities)
    audit = {
        "status": "passed",
        "training_only_selection_and_group_separation_verified": True,
        "selection_sha256": locked,
        "source_data_and_reference_identities_verified": True,
        "source_snapshot_files_verified": len(snapshot["files"]),
        "test_or_truth_used": False,
        "observations_checked_per_estimator": len(validation.event_ids),
        "estimators": records,
        "replayed_numerical_diagnostics": numerical,
        "report_source_sha256": study.shard_sha256(__file__),
        "renderer_source_sha256": study.shard_sha256(
            Path(__file__).with_name("neutrino_classical_visualize.py")
        ),
    }
    study.write_json(root / "evidence_audit.json", audit)
    return audit


def main():
    """Audit the finished classical comparison and export its local viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="data/neutrino/classical_comparison_v1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audit = audit_and_render(args.study, args.device)
    logging.getLogger(__name__).info(
        "Audit %s: %s estimators", audit["status"], len(audit["estimators"])
    )


if __name__ == "__main__":
    main()
