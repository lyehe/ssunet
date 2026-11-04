"""Replay a completed joint-detector checkpoint and export count/probability comparisons."""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..ssunet.confocal_inference import sha256_file, write_json
from ..ssunet.confocal_trial import evaluate_model, gaussian_probabilities
from ..ssunet.models.confocal_multichannel import MultichannelConfocalGAP, channel_log_probabilities
from .confocal_joint_train import (
    DEFAULT_CONFIG,
    convolution_hashes,
    load_config,
    prepare_data,
    source_hashes,
)


def _curve_figure(root, state, names, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    steps = [row["step"] for row in state["evaluations"]]
    for channel, name in enumerate(names):
        axes[0].plot(
            steps,
            [r["per_channel"][channel]["photon_nll"] for r in state["evaluations"]],
            "o-",
            label=name,
        )
    axes[0].set(xlabel="Optimizer update", ylabel="Reserved-photon NLL (nats/photon)")
    axes[0].legend()
    uniform = state["baselines"]["uniform"]["photon_nll"]
    axes[0].axhline(uniform, color="grey", linestyle="--", label="Uniform")
    scores = {"GAP4D": state["evaluations"][-1], **state["baselines"]}
    phases = list(state["evaluations"][-1]["per_phase"])
    x = np.arange(len(phases))
    for index, (method, score) in enumerate(scores.items()):
        axes[1].bar(
            x + (index - 1.5) * 0.2,
            [uniform - score["per_phase"][phase]["photon_nll"] for phase in phases],
            width=0.2,
            label=method,
        )
    axes[1].set(xticks=x, xticklabels=phases, ylabel="Gain over uniform (nats/photon)")
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Microtubule 3 | fixed reserved photons | higher gain is better")
    fig.savefig(root / "heldout_comparison.png", dpi=160)
    fig.savefig(root / "heldout_comparison.svg")
    plt.close(fig)


@torch.inference_mode()
def export_examples(root, model, cases, config, device):
    """Export predeclared first cases of all phases using the same predictions as scoring."""
    examples = []
    options = config["evaluation"]
    for phase in options["phases"]:
        case = next(case for case in cases if case["phase"] == phase)
        q = channel_log_probabilities(model(case["input"].to(device))).exp().cpu().numpy()[0]
        arrays = {
            "observed": case["observed"],
            "input": case["input"].numpy()[0],
            "heldout": case["target"].numpy()[0],
            "gap4d_probability": q,
        }
        for method, temporal in (
            ("gaussian3d", 0),
            ("gaussian4d", options["gaussian_temporal_sigma"]),
        ):
            arrays[f"{method}_probability"] = gaussian_probabilities(
                arrays["input"],
                [temporal, *options["gaussian_spatial_sigma_zyx"]],
                options["uniform_mixture"],
            )
        totals = case["observed"].sum(axis=(1, 2, 3, 4), dtype=np.int64)
        for method in ("gap4d", "gaussian3d", "gaussian4d"):
            arrays[f"{method}_expected_counts"] = (
                arrays[f"{method}_probability"] * totals[:, None, None, None, None]
            ).astype(np.float32)
        path = root / "examples" / phase
        path.mkdir(parents=True, exist_ok=True)
        for name, values in arrays.items():
            np.save(path / f"{name}.npy", values, allow_pickle=False)
        write_json(
            path / "metadata.json",
            {
                "case_index": case["index"],
                "phase": phase,
                "start_tzyx": case["start_tzyx"],
                "axis_order": "C,T,Z,Y,X",
                "observed_photons_per_detector": totals.tolist(),
                "scope": "Per-patch PDFs; display calibrated to each observed detector total",
                "clean_ground_truth": False,
            },
        )
        examples.append({"phase": phase, "arrays": arrays, "start": case["start_tzyx"]})
    return examples


def _image_figure(root, examples, names, plt, *, average):
    methods = [
        ("observed", "Observed (noisy)"),
        ("gap4d_expected_counts", "Joint GAP4D"),
        ("gaussian3d_expected_counts", "Gaussian 3D"),
        ("gaussian4d_expected_counts", "Gaussian 4D"),
    ]
    fields = []
    for example in examples:
        panels = {}
        for method, _ in methods:
            values = example["arrays"][method]
            z = values.shape[2] // 2
            panels[method] = (
                values[:, :, z].mean(axis=1) if average else values[:, values.shape[1] // 2, z]
            )
        fields.append(panels)
    maximum = [
        max(
            float(np.quantile(np.stack([p[key][c] for p in fields for key, _ in methods]), 0.997)),
            0.01,
        )
        for c in range(len(names))
    ]
    fig, axes = plt.subplots(
        len(examples) * len(names),
        len(methods),
        figsize=(13, 3.0 * len(examples) * len(names)),
        layout="constrained",
    )
    for phase_index, example in enumerate(examples):
        for channel, name in enumerate(names):
            for column, (method, label) in enumerate(methods):
                axis = axes[phase_index * len(names) + channel, column]
                artist = axis.imshow(
                    fields[phase_index][method][channel],
                    cmap="magma",
                    vmin=0,
                    vmax=maximum[channel],
                )
                axis.set_title(f"{example['phase']} | {name} | {label}", fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
                fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.02, label="counts/voxel")
    mode = "16-time mean" if average else "single time/Z slice"
    fig.suptitle(f"Microtubule 3 | {mode} | fixed scale per detector across phases | no clean GT")
    fig.savefig(
        root / ("time_mean_comparison.png" if average else "prediction_comparison.png"), dpi=145
    )
    plt.close(fig)


def _time_figure(root, names, plt):
    manifest = json.loads((root / "training_counts.json").read_text())
    totals = np.asarray(manifest["raw_and_training_photons_by_channel_time"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout="constrained")
    for channel, name in enumerate(names):
        for axis in axes:
            axis.plot(totals[channel, :, 0], label=name)
            axis.set(xlabel="Acquisition time point", ylabel="Observed photons in cached XY crop")
            axis.grid(alpha=0.2)
    axes[1].set_yscale("log")
    axes[0].legend()
    fig.suptitle("Raw counts in both detectors; declining signal is present in the observations")
    fig.savefig(root / "observed_time_course.png", dpi=160)
    plt.close(fig)


def _html_report(root, summary):
    images = "".join(
        '<img style="width:100%;height:auto" src="data:image/png;base64,'
        + base64.b64encode((root / name).read_bytes()).decode("ascii")
        + '">'
        for name in (
            "heldout_comparison.png",
            "observed_time_course.png",
            "prediction_comparison.png",
            "time_mean_comparison.png",
        )
    )
    compact = {
        method: {
            key: score[key]
            for key in ("photon_nll", "channel_mean_nll", "per_channel", "per_phase")
        }
        for method, score in summary["comparison"].items()
    }
    document = (
        '<!doctype html><html><meta charset="utf-8">'
        "<title>Microtubule 3: joint-channel 4D GAP</title>"
        '<body style="font:16px system-ui;max-width:1400px;margin:24px auto;padding:0 16px">'
        "<h1>Microtubule 3: both detector channels</h1>"
        "<p>500-update single-acquisition pilot. Evaluation photons were permanently excluded "
        "from training. Predictions are per-detector arrival distributions; expected-count "
        "images use each detector's observed patch total. No clean ground truth or PSNR.</p>"
        "<p>Gaussian settings were fixed before fitting. The neural model has cross-detector "
        "context; these comparisons alone cannot attribute a difference "
        "to temporal convolution.</p>"
        "<pre>" + json.dumps(compact, indent=2) + "</pre>" + images + "</body></html>"
    )
    (root / "index.html").write_text(document, encoding="utf-8")


def main():
    """Independently replay the final checkpoint before exporting its visual comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = load_config(args.config)
    root = Path(config["output"])
    result = json.loads((root / "training/result.json").read_text())
    expected = result["identity"]
    if (
        source_hashes() != expected["source_hashes"]
        or config != expected["config"]
        or convolution_hashes() != expected["convolution"]
    ):
        message = "Trial sources or configuration changed before replay"
        raise ValueError(message)
    for name, digest in expected["source_hashes"].items():
        if sha256_file(root / "source_snapshot" / name) != digest:
            message = "Archived trial source changed before replay"
            raise ValueError(message)
    pool, cases, manifests = prepare_data(config)
    if (
        manifests["raw"]["sha256"] != expected["raw_sha256"]
        or manifests["training_pool"]["sha256"] != expected["training_pool_sha256"]
    ):
        message = "Raw data or training photon pool changed before replay"
        raise ValueError(message)
    path = root / "training/last.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint["identity"] != expected
        or checkpoint["state"]["step"] != config["training"]["steps"]
    ):
        message = "Checkpoint identity or fixed endpoint does not match the completed trial"
        raise ValueError(message)
    device = torch.device("cuda:0")
    model = MultichannelConfocalGAP(channels=pool.shape[0], **config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    replay = evaluate_model(model, cases, device)
    stored = result["state"]["evaluations"][-1]
    difference = abs(replay["photon_nll"] - stored["photon_nll"])
    for a, b in zip(replay["cases"], stored["cases"], strict=True):
        for c, d in zip(a["channels"], b["channels"], strict=True):
            if c["target_photons"] != d["target_photons"] or not math.isclose(
                c["total_nll"], d["total_nll"], rel_tol=1e-7, abs_tol=1e-6
            ):
                message = "Checkpoint per-case likelihood did not replay"
                raise ValueError(message)
    if difference > 1e-6:
        message = "Checkpoint likelihood did not replay"
        raise ValueError(message)
    examples = export_examples(root, model, cases, config, device)
    names = expected["channel_names"]
    _curve_figure(root, result["state"], names, plt)
    _image_figure(root, examples, names, plt, average=False)
    _image_figure(root, examples, names, plt, average=True)
    _time_figure(root, names, plt)
    summary = {
        "status": "passed",
        "scope": config["evaluation"]["scope"],
        "checkpoint_sha256": sha256_file(path),
        "replay_nll_difference": difference,
        "max_probability_mass_error": replay["max_probability_mass_error"],
        "comparison": {"joint_gap4d": replay, **result["state"]["baselines"]},
        "clean_ground_truth": False,
    }
    write_json(root / "comparison.json", summary)
    _html_report(root, summary)


if __name__ == "__main__":
    main()
