"""Render live confocal training status and audit the completed fixed comparison."""

from __future__ import annotations

import argparse
import base64
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def render_live(study):
    """Export a local dashboard that reads atomically published worker progress."""
    config = yaml.safe_load((study / "config.yml").read_text())
    models = {}
    for spec in config["models"]:
        batching = dict(config["training"])
        batching.update(batching.get("model_batching", {}).get(spec["name"], {}))
        # Parameter counts below were checked against the instantiated backbones.
        parameters = {"conv4d": 17_973_665, "conv3d": 17_797_591}[spec["name"]]
        models[spec["name"]] = {
            "widths": [spec["base_channels"] * 2**level for level in range(spec["depth"])],
            "parameters": parameters,
            "batch_size": batching["batch_size"],
            "accumulation_steps": batching["accumulation_steps"],
        }
    metadata = {
        "models": models,
        "uniform_nll": math.log(math.prod(config["training"]["patch_tzyx"])),
        "setup": {
            "Patch (T x Z x Y x X)": " x ".join(map(str, config["training"]["patch_tzyx"])),
            "Training": "double snr data 2.lif · Series001 · channel index 1",
            "Validation data": "XYT cell microtuble 4.lif · Series013 · channel index 1",
            "Photon supervision": "GAP · raw integer counts · p = 0.5 · ten sine features",
            "Logit normalization": "Barron et al. (2026), fixed log-softmax, equation 5",
            "Training budget": "500 Adam updates x 4 patches, per model · one seed (42)",
            "Scoring": "16 fixed patches from a separate acquisition; final fixed-step endpoint",
            "Hardware": "Physical GPU 0 · RTX 4090 · FP32 · TF32 disabled",
        },
    }
    _json(study / "study_metadata.json", metadata)
    paths = ["study_metadata.json", "training/pilot/process.json", "comparison_summary.json"]
    for name in ("conv4d", "conv3d"):
        paths.extend(
            f"training/pilot/{name}/{file}"
            for file in ("progress.json", "learning_curve.json", "result.json")
        )
    snapshot = {p: json.loads((study / p).read_text()) for p in paths if (study / p).exists()}
    snapshot["generated_utc"] = datetime.now(timezone.utc).isoformat()
    template = Path(__file__).with_name("confocal_results.html").read_text(encoding="utf-8")
    for filename in ("observations.png", "prediction_examples.png"):
        path = study / filename
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            template = template.replace(
                f'src="{filename}"', f'src="data:image/png;base64,{encoded}"'
            )
    embedded = json.dumps(snapshot).replace("<", "\\u003c")
    (study / "index.html").write_text(
        template.replace("__OFFLINE_DATA__", embedded), encoding="utf-8"
    )
    return config


def render_observations(study, config):
    """Show the predeclared validation patches without loading a model or a GPU."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..ssunet.datasets.confocal import sample_split

    counts = np.load(Path(config["cache"]) / "validation.npy", mmap_mode="r", allow_pickle=False)
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), layout="constrained")
    for row, index in enumerate((0, 7, 15)):
        retained, target = sample_split(
            counts,
            config["training"]["patch_tzyx"],
            batch_size=1,
            p=config["split_probability"],
            seed=config["photon_split_seed"],
            index=index,
            stream=1,
        )
        a, b = retained.numpy()[0, 0], target.numpy()[0, 0]
        t, z = a.shape[0] // 2, a.shape[1] // 2
        reference = (a.astype(np.float32) + b).mean(axis=0)[z]
        for col, (array, title) in enumerate(
            (
                (a[t, z], "Input A | counts"),
                (b[t, z], "Held-out B | counts"),
                (reference, "16-volume mean C | counts (noisy)"),
            )
        ):
            vmax = 3 if col < 2 else max(float(np.quantile(reference, 0.997)), 0.1)
            artist = axes[row, col].imshow(
                array, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest"
            )
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            fig.colorbar(artist, ax=axes[row, col], fraction=0.046, pad=0.02)
        axes[row, 0].set_ylabel(f"Patch {index}\nT={t}, Z={z}", fontsize=10)
    fig.suptitle("Real HyD photon-counting observations | no clean ground truth", fontsize=14)
    fig.savefig(study / "observations.png", dpi=150)
    plt.close(fig)


def softmax_audit(study):
    """Record the primary paper's tiny-gradient example against analytic values."""
    import torch

    from ..ssunet.distributions import logits_to_log_probabilities
    from ..ssunet.softmax_fix import fixed_log_softmax

    report = {
        "reference": "https://jonbarron.info/softmaxfix/",
        "equation": 5,
        "scope": "Fixed log-softmax for observed photon-count targets; no teacher logits",
        "cases": [],
    }
    for dtype, peak in ((torch.float32, 9.0), (torch.float64, 20.0)):
        row = {"dtype": str(dtype), "logits": [peak, -peak, -peak]}
        for name, function in (
            ("pytorch_log_softmax", lambda x: x.log_softmax(-1)),
            ("previous_explicit_centering", lambda x: logits_to_log_probabilities(x[None])[0]),
            ("barron_fixed", fixed_log_softmax),
        ):
            logits = torch.tensor(row["logits"], dtype=dtype, requires_grad=True)
            loss = -function(logits)[0]
            row[name] = {
                "loss": float(loss.detach()),
                "gradient": torch.autograd.grad(loss, logits)[0].tolist(),
            }
        tail = 2 * math.exp(-2 * peak)
        row["analytic_loss"] = math.log1p(tail)
        row["analytic_dominant_gradient"] = -tail / (1 + tail)
        report["cases"].append(row)
    report["note"] = (
        "Previous explicit max-centering preserves the dominant gradient in these examples, "
        "but rounds the tiny forward log probability to zero. The new formula preserves both."
    )
    _json(study / "softmax_audit.json", report)


def render_curves(study):
    """Save standard exportable scientific plots from the recorded learning curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for name, label, color in (("conv4d", "4D GAP", "#2563eb"), ("conv3d", "3D GAP", "#c2410c")):
        path = study / "training/pilot" / name / "learning_curve.json"
        if not path.exists():
            continue
        curve = json.loads(path.read_text())
        for axis, metric in zip(axes, ("photon_nll", "patch_mean_nll"), strict=True):
            axis.plot(
                [x["step"] for x in curve],
                [x[metric] for x in curve],
                "o-",
                label=label,
                color=color,
            )
    for axis in axes:
        axis.axhline(math.log(4_194_304), color="gray", ls="--", lw=1, label="Uniform")
        axis.set_xlabel("Optimizer update")
        axis.set_ylabel("Held-out NLL (nats/photon) ↓")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_title("Photon-weighted")
    axes[1].set_title("Equal nonempty-patch weighting")
    fig.savefig(study / "learning_curves.png", dpi=170)
    fig.savefig(study / "learning_curves.svg")
    plt.close(fig)


def main():
    """Prepare a live report or refresh its recorded scientific plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=Path("data/confocal_gap4d_features32_batch4_v1")
    )
    parser.add_argument("--observations", action="store_true")
    parser.add_argument("--softmax-audit", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((args.study / "config.yml").read_text())
    render_curves(args.study)
    if args.observations:
        render_observations(args.study, config)
    if args.softmax_audit:
        softmax_audit(args.study)
    render_live(args.study)


if __name__ == "__main__":
    main()
