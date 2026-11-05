"""Inspect every fixed scored patch without selecting favorable image examples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..ssunet.confocal_display import detector_matplotlib_colormap
from ..ssunet.confocal_inference import sha256_file, write_json
from ..ssunet.confocal_trial import gaussian_probabilities, validation_cases
from ..ssunet.models.confocal_multichannel import MultichannelConfocalGAP, channel_log_probabilities
from .confocal_joint_train import DEFAULT_CONFIG, load_config, source_hashes


@torch.inference_mode()
def export_cases(root, config, model, cases):
    """Retain all cases, validating their likelihoods against the completed primary audit."""
    audit = json.loads((root / "comparison.json").read_text())
    records, projections = [], []
    options = config["evaluation"]
    for case, stored in zip(cases, audit["comparison"]["joint_gap4d"]["cases"], strict=True):
        logq = channel_log_probabilities(model(case["input"].to("cuda:0")))
        observed = case["observed"]
        totals = observed.sum(axis=(1, 2, 3, 4), dtype=np.int64)
        target = case["target"].to("cuda:0")
        nll = (-(target.double() * logq.double()).flatten(2).sum(-1))[0].tolist()
        for c, row in enumerate(stored["channels"]):
            if abs(nll[c] - row["total_nll"]) > 1e-6 * max(row["target_photons"], 1):
                message = "All-case visualization did not reproduce the scored likelihood"
                raise ValueError(message)
        q = logq.exp().cpu().numpy()[0]
        expected = (q * totals[:, None, None, None, None]).astype(np.float32)
        smooth = gaussian_probabilities(
            case["input"].numpy()[0],
            [options["gaussian_temporal_sigma"], *options["gaussian_spatial_sigma_zyx"]],
            options["uniform_mixture"],
        )
        smooth = (smooth * totals[:, None, None, None, None]).astype(np.float32)
        directory = f"{case['phase']}_case{case['index']:02d}"
        destination = root / "examples" / directory
        destination.mkdir(parents=True, exist_ok=True)
        for name, values in (
            ("observed", observed),
            ("gap4d_probability", q),
            ("gap4d_expected_counts", expected),
        ):
            np.save(destination / f"{name}.npy", values, allow_pickle=False)
        metadata = {
            "case_index": case["index"],
            "phase": case["phase"],
            "start_tzyx": case["start_tzyx"],
            "axis_order": "C,T,Z,Y,X",
            "observed_photons_per_detector": totals.tolist(),
            "clean_ground_truth": False,
        }
        write_json(destination / "metadata.json", metadata)
        records.append({"directory": directory, **metadata})
        projections.append(
            np.stack([a.mean(axis=(1, 2), dtype=np.float32) for a in (observed, expected, smooth)])
        )
    write_json(
        root / "examples/index.json",
        {
            "scope": "Every fixed evaluation case, same checkpoint and inputs as primary scoring",
            "cases": records,
        },
    )
    return records, np.stack(projections)


def plot_cases(root, records, fields, names):
    """Show time-and-depth means on a common detector scale across all cases and phases."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maximum = [max(float(np.quantile(fields[:, :, c], 0.997)), 0.01) for c in range(len(names))]
    methods = ("Observed, noisy", "Joint GAP4D", "Gaussian 4D")
    for phase in dict.fromkeys(row["phase"] for row in records):
        indices = [i for i, record in enumerate(records) if record["phase"] == phase]
        fig, axes = plt.subplots(
            len(indices) * len(names),
            3,
            figsize=(11, 2.8 * len(indices) * len(names)),
            layout="constrained",
        )
        for row, index in enumerate(indices):
            record = records[index]
            for channel, name in enumerate(names):
                for method, label in enumerate(methods):
                    axis = axes[row * len(names) + channel, method]
                    artist = axis.imshow(
                        fields[index, method, channel],
                        cmap=detector_matplotlib_colormap(name),
                        vmin=0,
                        vmax=maximum[channel],
                    )
                    axis.set_title(f"Case {record['case_index']} | {name} | {label}", fontsize=9)
                    axis.set_xticks([])
                    axis.set_yticks([])
                    fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.02)
        fig.suptitle(
            f"{phase} | every fixed case | mean over 16 times and 16 Z planes\n"
            "Counts/voxel; same scale per detector across phases; no clean GT"
        )
        fig.savefig(root / f"all_cases_{phase}.png", dpi=145)
        plt.close(fig)


def main():
    """Extend the visual inspection to all scored cases after the primary replay passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["output"])
    result = json.loads((root / "training/result.json").read_text())
    audit = json.loads((root / "comparison.json").read_text())
    checkpoint_path = root / "training/last.pt"
    if (
        audit["status"] != "passed"
        or sha256_file(checkpoint_path) != audit["checkpoint_sha256"]
        or source_hashes() != result["identity"]["source_hashes"]
    ):
        message = "Use the unchanged checkpoint and sources from the completed primary audit"
        raise ValueError(message)
    raw = np.load(root / "counts.npy", mmap_mode="r", allow_pickle=False)
    pool = np.load(root / "training_counts.npy", mmap_mode="r", allow_pickle=False)
    cases = validation_cases(raw, pool, config)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = MultichannelConfocalGAP(channels=raw.shape[0], **config["model"]).to("cuda:0")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    records, projections = export_cases(root, config, model, cases)
    plot_cases(root, records, projections, result["identity"]["channel_names"])
    write_json(
        root / "all_cases_visual_audit.json",
        {
            "status": "passed",
            "cases": len(cases),
            "checkpoint_sha256": audit["checkpoint_sha256"],
            "source_sha256": sha256_file(Path(__file__)),
            "scope": "Supplemental inspection of every scored case; primary protocol unchanged",
        },
    )


if __name__ == "__main__":
    main()
