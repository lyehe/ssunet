"""Replay the completed confocal comparison and export distribution predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _validate_sources(study, expected):
    for name, digest in expected.items():
        snapshot = study / "source_snapshot" / name
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            message = f"Current source differs from the trained implementation: {name}"
            raise ValueError(message)
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            message = f"Archived source has changed: {name}"
            raise ValueError(message)


def _validate_external_package(study):
    record = json.loads((study / "environment.json").read_text())
    package = Path(importlib.util.find_spec("torch_convnd").origin).parent
    for name, expected in record["package_source_sha256"].items():
        current = hashlib.sha256((package / name).read_bytes()).hexdigest()
        archived = hashlib.sha256(
            (study / "environment/torch_convnd" / name).read_bytes()
        ).hexdigest()
        if current != expected or archived != expected:
            message = f"Convolution package source differs: {name}"
            raise ValueError(message)


def _prediction_figure(study, arrays):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 5, figsize=(16, 9), layout="constrained")
    for row, index in enumerate(arrays["example_indices"]):
        a, b = arrays["input"][row], arrays["target"][row]
        t, z = a.shape[0] // 2, a.shape[1] // 2
        three = arrays["conv3d"][row, t, z] * 1e6
        four = arrays["conv4d"][row, t, z] * 1e6
        reference = (a.astype(np.float32) + b).mean(axis=0)[z]
        probability_max = max(float(np.quantile(np.stack([three, four]), 0.997)), 0.1)
        fields = [a[t, z], b[t, z], three, four, reference]
        titles = [
            "Input A | counts",
            "Held-out B | counts",
            "3D q per voxel (x1e6)",
            "4D q per voxel (x1e6)",
            "Temporal mean C (noisy)",
        ]
        for col, (field, title) in enumerate(zip(fields, titles, strict=True)):
            vmax = 3 if col < 2 else probability_max
            if col == 4:
                vmax = max(float(np.quantile(reference, 0.997)), 0.1)
            artist = axes[row, col].imshow(
                field, vmin=0, vmax=vmax, cmap="magma", interpolation="nearest"
            )
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            fig.colorbar(artist, ax=axes[row, col], fraction=0.046, pad=0.02)
        axes[row, 0].set_ylabel(f"Patch {index}\nT={t}, Z={z}", fontsize=10)
    fig.suptitle(
        "Held-out confocal observations and predicted arrival probabilities\n"
        "Temporal means are noisy context, not clean ground truth",
        fontsize=13,
    )
    fig.savefig(study / "prediction_examples.png", dpi=160)
    fig.savefig(study / "prediction_examples.svg")
    plt.close(fig)


def _check_scores(actual, expected):
    errors = {}
    for key in ("photon_nll", "patch_mean_nll", "per_patch_total_nll"):
        if not np.isfinite(actual[key]).all() or not np.isfinite(expected[key]).all():
            message = "Nonfinite likelihood in checkpoint replay"
            raise ValueError(message)
        errors[key] = float(np.max(np.abs(np.asarray(actual[key]) - np.asarray(expected[key]))))
    if not np.array_equal(actual["per_patch_target_photons"], expected["per_patch_target_photons"]):
        message = "Validation photon draws differ from the recorded run"
        raise ValueError(message)
    if errors["photon_nll"] > 1e-6 or errors["per_patch_total_nll"] > 0.5:
        message = "Checkpoint likelihood does not replay within floating-point tolerance"
        raise ValueError(message)
    if actual["max_probability_mass_error"] > 1e-5:
        message = "Predicted probability mass does not normalize"
        raise ValueError(message)
    return errors


def _comparison_csv(study, models):
    columns = [
        "model",
        "parameters",
        "batch_size",
        "accumulation_steps",
        "effective_batch_size",
        "optimizer_steps",
        "training_examples",
        "photon_nll",
        "patch_mean_nll",
        "training_seconds",
        "peak_gpu_allocated_gib",
        "peak_gpu_reserved_gib",
    ]
    with (study / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, values in models.items():
            result = json.loads((study / "training/pilot" / name / "result.json").read_text())
            row = {key: result[key] for key in columns if key != "model" and key in result}
            row.update(
                model=name,
                photon_nll=values["scores"]["photon_nll"],
                patch_mean_nll=values["scores"]["patch_mean_nll"],
            )
            writer.writerow(row)


def audit(study):
    """Require completed training before allocating another model on GPU 0."""
    process = json.loads((study / "training/pilot/process.json").read_text())
    if process["status"] != "complete":
        message = "Wait for the full comparison to finish before GPU checkpoint replay"
        raise RuntimeError(message)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch

    from ..ssunet.models.confocal_gap import ConfocalGAPUNet
    from ..ssunet.softmax_fix import fixed_log_softmax
    from .confocal_train import (
        evaluate,
        fixed_validation,
        load_config,
        prepare_data,
        resolve_model_config,
    )

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_config(study / "config.yml")
    counts, manifests = prepare_data(config)
    validation = fixed_validation(counts["validation"], config)
    source_hashes = json.loads((study / "source_snapshot.json").read_text())
    _validate_sources(study, source_hashes)
    _validate_external_package(study)
    device = torch.device("cuda:0")
    report, final = {}, {}
    examples = np.asarray([0, 7, 15])
    arrays = {
        "example_indices": examples,
        "input": np.stack([validation[i][0].numpy()[0, 0].astype(np.uint8) for i in examples]),
        "target": np.stack([validation[i][1].numpy()[0, 0].astype(np.uint8) for i in examples]),
    }
    for settings in config["models"]:
        name = settings["name"]
        directory = study / "training/pilot" / name
        identity = json.loads((directory / "identity.json").read_text())
        if identity["source_hashes"] != source_hashes:
            message = "The model arms used different source versions"
            raise ValueError(message)
        if identity["sources"] != manifests or identity["config"] != resolve_model_config(
            config, name
        ):
            message = "Frozen source/crop configuration does not match the completed run"
            raise ValueError(message)
        saved = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        result = json.loads((directory / "result.json").read_text())
        if (
            saved["step"] != config["training"]["steps"]
            or result["optimizer_steps"] != saved["step"]
        ):
            message = "A model did not finish the complete fixed schedule"
            raise ValueError(message)
        model = ConfocalGAPUNet(**{k: v for k, v in settings.items() if k != "name"}).to(device)
        model.load_state_dict(saved["model_state_dict"])
        scores = evaluate(model, validation, device, normalization="barron2026")
        errors = _check_scores(scores, result["curve"][-1])
        probabilities = []
        with torch.inference_mode():
            for index in examples:
                logits = model(validation[index][0].to(device))
                q = fixed_log_softmax(logits.flatten(1)).exp().reshape_as(logits)
                probabilities.append(q[0, 0].cpu().numpy())
        arrays[name] = np.stack(probabilities)
        report[name] = {
            "score_replay_error": errors,
            "max_probability_mass_error": scores["max_probability_mass_error"],
            "checkpoint_sha256": hashlib.sha256((directory / "last.pt").read_bytes()).hexdigest(),
            "optimizer_steps": saved["step"],
            "effective_batch_size": result["effective_batch_size"],
            "training_examples": result["training_examples"],
            "logit_normalization": result["logit_normalization"],
        }
        final[name] = {
            "scores": scores,
            "training_seconds": result["training_seconds"],
            "parameters": result["parameters"],
            "peak_gpu_allocated_gib": result["peak_gpu_allocated_gib"],
            "peak_gpu_reserved_gib": result["peak_gpu_reserved_gib"],
        }
        del model, saved
        torch.cuda.empty_cache()
    delta = final["conv3d"]["scores"]["photon_nll"] - final["conv4d"]["scores"]["photon_nll"]
    summary = {
        "models": final,
        "three_minus_four_nats_per_photon": delta,
        "description": (
            f"Final held-out photon NLL: 4D {final['conv4d']['scores']['photon_nll']:.5f}; "
            f"3D {final['conv3d']['scores']['photon_nll']:.5f}. "
            f"3D minus 4D = {delta:.5f} nats/photon. "
            "This is a single-seed acquisition-holdout pilot; validation patches may overlap."
        ),
        "scope": config["evaluation"]["scope"],
    }
    np.savez_compressed(study / "prediction_examples.npz", **arrays)
    _prediction_figure(study, arrays)
    _write(
        study / "evidence_audit.json",
        {
            "status": "passed",
            "models": report,
            "source_snapshot_verified": True,
            "cache_identity_verified": True,
            "external_convolution_source_verified": True,
        },
    )
    _write(study / "comparison_summary.json", summary)
    _comparison_csv(study, final)
    return summary


def main():
    """Audit completed checkpoints and refresh the standalone report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=Path("data/confocal_gap4d_features32_batch4_v1")
    )
    args = parser.parse_args()
    audit(args.study)
    from .confocal_report import render_curves, render_live

    render_curves(args.study)
    render_live(args.study)


if __name__ == "__main__":
    main()
