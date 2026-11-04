"""Train on analytic Poisson observations and evaluate against known location PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml
from torch.utils.data import DataLoader

from ..ssunet.datasets.neutrino import NeutrinoPhotonDataset, shard_sha256
from ..ssunet.models.neutrino_dense import DensePhotonReference
from ..ssunet.neutrino_evaluation import photon_score_summary
from ..ssunet.neutrino_streaming import (
    PhotonBaselineBank,
    collect_training_reference,
    select_baselines,
    stream_scores,
)
from ..ssunet.neutrino_training import train_photon_seed
from ..ssunet.photon_reference import photon_count_log_probabilities
from .neutrino_prepare import write_json
from .neutrino_reference import create_analytic_reference
from .neutrino_study import _atomic_scores, _code_hashes, _score_bundle

LOGGER = logging.getLogger(__name__)


def analytic_location_pdf(positions, edges, source_position, signal, background):
    """Reconstruct the exact discrete generator rate for evaluation only."""
    centers = (edges[:-1] + edges[1:]) / 2
    distance = np.linalg.norm(positions - source_position, axis=1)
    delay = centers[None, :] - (500 + distance / 0.225)[:, None]
    temporal = np.exp(-0.5 * (delay / 60) ** 2)
    temporal += 0.35 * np.exp(-np.maximum(delay, 0) / 200) * (delay >= 0)
    spatial = np.exp(-distance / 300) / (distance**2 + 50**2)
    signal_pdf = spatial[:, None] * temporal
    signal_pdf /= signal_pdf.sum()
    intensity = signal * signal_pdf + background / signal_pdf.size
    if intensity.sum() <= 0:
        message = "A zero-intensity state has no conditional photon-location distribution"
        raise ValueError(message)
    return intensity / intensity.sum()


def open_analytic(config, partition, *, training=False, selected=None):
    """Open observations without exposing the evaluation-only source parameters."""
    return NeutrinoPhotonDataset(
        Path(config["data"]["output"]) / "observations.h5",
        partition,
        lazy=True,
        p=config["data"]["split_probability"],
        seed=config["data"]["photon_split_seed"],
        training=training,
        selected_event_ids=selected,
    )


def _baseline_bank(config, training, reference, device, selected=None):
    """Build controls exclusively from observed training detections."""
    bank = PhotonBaselineBank(
        reference["observed_counts"],
        training.positions_m,
        training.live,
        training.edges_ns,
        config["baselines"],
        device,
    )
    if selected is not None:
        bank.selected_prior_parameters = selected["selected_prior"]
        bank.selected_mixture_parameters = selected["selected_mixture"]
        bank.winner = selected["selected_method"]
        bank._set_priors()
    return bank


def _identity(config, observations):
    """Pin observations and implementation before accessing any test outcomes."""
    code = _code_hashes()
    code[str(Path(__file__).resolve())] = shard_sha256(Path(__file__))
    return {
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "observations_sha256": shard_sha256(observations),
        "code_sha256": code,
    }


def train_analytic(config, device):
    """Fit three unchanged backbones using only analytic training A/B splits."""
    root = Path(config["data"]["output"])
    root.mkdir(parents=True, exist_ok=True)
    observations = root / "observations.h5"
    if not observations.exists():
        inspection = create_analytic_reference(observations, config["generator"])
        write_json(root / "observations.inspection.json", inspection)
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    training = open_analytic(config, "train")
    validation = open_analytic(config, "validation")
    reference = collect_training_reference(training)
    if (root / "baseline_selection.json").exists():
        selected = json.loads((root / "baseline_selection.json").read_text())
        bank = _baseline_bank(config, training, reference, device, selected)
    else:
        bank = _baseline_bank(config, training, reference, device)
        selected = select_baselines(reference, validation, bank, config["training"]["batch_size"])
        write_json(root / "baseline_selection.json", selected)
    records, values = stream_scores(
        validation, batch_size=config["training"]["batch_size"], device=device, baselines=bank
    )
    _atomic_scores(root / "validation_baseline_scores.npz", records, values)
    selected_ids = training.event_ids[training.indices][
        : config["evaluation"]["fixed_training_diagnostic_events"]
    ]
    diagnostic = open_analytic(config, "train", selected=selected_ids)
    fitting = open_analytic(config, "train", training=True)
    identity = _identity(config, observations)
    manifest = root / "training_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
        message = "Analytic study identity differs from its recorded protocol"
        raise ValueError(message)
    write_json(
        manifest, {"identity": identity, "test_scored": False, "truth_used_by_training": False}
    )
    results = []
    for seed in config["training"]["seeds"]:
        result = train_photon_seed(
            config,
            seed,
            (fitting, validation, diagnostic),
            lambda: DensePhotonReference(config["model"]["channels"]),
            root / f"runs/seed_{seed}",
            device,
            identity,
        )
        results.append(result)
        if not result["completed_fixed_epochs"]:
            write_json(root / "development_progress.json", {"runs": results})
            return
    lock = {
        "identity": identity,
        "checkpoints": {
            str(seed): {
                "path": str(root / f"runs/seed_{seed}/best_low_count.pt"),
                "sha256": shard_sha256(root / f"runs/seed_{seed}/best_low_count.pt"),
            }
            for seed in config["training"]["seeds"]
        },
        "baseline_selection_sha256": shard_sha256(root / "baseline_selection.json"),
        "evaluation_truth_sha256": shard_sha256(observations.with_suffix(".evaluation_only.npz")),
        "runs": results,
        "known_pdf_role": "evaluation_only_oracle_with_extra_latent_information",
    }
    if not (root / "analysis_lock.json").exists():
        write_json(root / "analysis_lock.json", lock)
    for dataset in (training, validation, diagnostic, fitting):
        dataset.close()


@torch.inference_mode()
def evaluate_oracle_model(model, dataset, oracle_sources, bank, config, device):
    """Compute latent-oracle NLL, KL and eventwise errors without training on GT."""
    size = len(dataset)
    nll, oracle_nll, kl, baseline_kl, expected = (np.zeros(size) for _ in range(5))
    by_level = np.zeros(size)
    offset = 0
    levels = config["generator"]["analytic"]["expected_signal_photons"]
    background = config["generator"]["analytic"]["expected_background_photons"]
    model.eval()
    for batch in DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False):
        counts, target = batch["input"].to(device), batch["target"].to(device)
        logits = model(
            counts, batch["positions_m"].to(device), batch["live"].to(device), batch["p"].to(device)
        )
        logq = (
            photon_count_log_probabilities(logits, batch["live"].to(device)).cpu().double().numpy()
        )
        baseline = bank.probabilities(counts)[bank.winner].cpu().double().numpy()
        for local, event in enumerate(batch["event_id"].numpy()):
            group = int(event) // len(levels)
            signal = levels[int(event) % len(levels)]
            pdf = analytic_location_pdf(
                dataset.positions_m, dataset.edges_ns, oracle_sources[group], signal, background
            )
            truth_log = np.log(pdf)
            index = offset + local
            b = target[local].cpu().numpy()
            nll[index] = -(b * logq[local]).sum()
            oracle_nll[index] = -(b * truth_log).sum()
            kl[index] = (pdf * (truth_log - logq[local])).sum()
            baseline_kl[index] = (pdf * (truth_log - np.log(baseline[local]))).sum()
            expected[index], by_level[index] = signal + background, signal
        offset += len(counts)
    return {
        "model": nll,
        "latent_oracle": oracle_nll,
        "model_kl": kl,
        "baseline_kl": baseline_kl,
        "expected_photons": expected,
        "signal_level": by_level,
    }


def test_analytic(config, device):
    """Evaluate the sealed analytic comparison once, including true PDF diagnostics."""
    root = Path(config["data"]["output"])
    if (root / "test_results.json").exists():
        message = "Analytic test results already exist; preserve their first evaluation"
        raise FileExistsError(message)
    lock = json.loads((root / "analysis_lock.json").read_text())
    if _identity(config, root / "observations.h5") != lock["identity"]:
        message = "Analytic implementation or observations changed after model selection"
        raise ValueError(message)
    for item in lock["checkpoints"].values():
        if shard_sha256(item["path"]) != item["sha256"]:
            message = "An analytic selected checkpoint changed after selection"
            raise ValueError(message)
    truth_path = root / "observations.evaluation_only.npz"
    if shard_sha256(truth_path) != lock["evaluation_truth_sha256"]:
        message = "Evaluation-only analytic parameters changed"
        raise ValueError(message)
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    training, testing = open_analytic(config, "train"), open_analytic(config, "test")
    reference = collect_training_reference(training)
    selected = json.loads((root / "baseline_selection.json").read_text())
    bank = _baseline_bank(config, training, reference, device, selected)
    records, baseline_values = stream_scores(
        testing, batch_size=config["training"]["batch_size"], device=device, baselines=bank
    )
    started = time.perf_counter()
    write_json(
        root / "test_opened.json",
        {
            "analysis_lock_sha256": shard_sha256(root / "analysis_lock.json"),
            "pid": os.getpid(),
            "process_created": psutil.Process().create_time(),
        },
    )
    runs, diagnostics = [], []
    with np.load(truth_path, allow_pickle=False) as truth:
        sources = truth["latent_source_position_m"]
    for seed in config["training"]["seeds"]:
        checkpoint = torch.load(
            lock["checkpoints"][str(seed)]["path"], map_location="cpu", weights_only=True
        )
        model = DensePhotonReference(config["model"]["channels"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        values = evaluate_oracle_model(model, testing, sources, bank, config, device)
        _atomic_scores(root / f"test_seed_{seed}.npz", records, values)
        diagnostics.append(values)
        runs.append(
            {
                "seed": seed,
                "scores": photon_score_summary(values["model"], records),
                "mean_event_KL_to_known_pdf": float(values["model_kl"].mean()),
            }
        )
    values = dict(baseline_values)
    values["model"] = np.mean([d["model"] for d in diagnostics], axis=0)
    values["latent_oracle"] = diagnostics[0]["latent_oracle"]
    result = _score_bundle(records, values, config, bank.winner)
    result.update(
        {
            "scope": "independent_analytic_test; known_oracle_has_unobserved_latent_source",
            "runs": runs,
            "elapsed_seconds": time.perf_counter() - started,
            "truth_used_by_training": False,
            "test_groups": len(np.unique(records.configuration_ids)),
            "mean_seed_event_KL_to_known_pdf": float(
                np.mean([d["model_kl"].mean() for d in diagnostics])
            ),
            "mean_event_baseline_KL_to_known_pdf": float(diagnostics[0]["baseline_kl"].mean()),
            "by_expected_signal": {
                str(level): {
                    "events": int((diagnostics[0]["signal_level"] == level).sum()),
                    "mean_seed_event_model_KL": float(
                        np.mean(
                            [d["model_kl"][d["signal_level"] == level].mean() for d in diagnostics]
                        )
                    ),
                    "mean_event_baseline_KL": float(
                        diagnostics[0]["baseline_kl"][
                            diagnostics[0]["signal_level"] == level
                        ].mean()
                    ),
                }
                for level in config["generator"]["analytic"]["expected_signal_photons"]
            },
        }
    )
    _atomic_scores(root / "test_mean_seed_scores.npz", records, values)
    write_json(root / "test_results.json", result)
    training.close()
    testing.close()


def main():
    """Run a frozen analytic training or first test-evaluation stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "test"))
    parser.add_argument("--config", type=Path, default=Path("config/neutrino_diagnostics.yml"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = yaml.safe_load(args.config.read_text())
    if args.action == "train":
        train_analytic(config, args.device)
    else:
        test_analytic(config, args.device)


if __name__ == "__main__":
    main()
