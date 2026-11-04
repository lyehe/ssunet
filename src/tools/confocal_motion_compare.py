"""Compare motion preservation and photon likelihoods for both two-detector recordings."""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import math
from pathlib import Path
from string import Template

import numpy as np
import yaml
from PIL import Image, ImageDraw

from ..ssunet.confocal_inference import sha256_file, write_json
from ..ssunet.confocal_motion import split_motion_pool
from ..ssunet.confocal_motion_comparison import (
    factorized_motion_probability,
    gaussian_motion_probability,
    photon_log_score,
)
from .confocal_motion_report import _fixed_palette, block_trajectories
from .confocal_motion_review import select_region

LOGGER = logging.getLogger(__name__)
METHODS = {
    "gaussian3d": "Spatial Gaussian",
    "gaussian4d": "Space + time Gaussian",
    "gap4d": "Two-channel GAP",
    "gap4d_frozen": "GAP frozen spatial shape",
    "uniform": "Uniform",
}
DISPLAY = ("observed", "gaussian3d", "gaussian4d", "gap4d")


def _check_sources(study, output, config):
    index = json.loads((study / "index.json").read_text())
    if index["status"] != "complete":
        message = "Complete the continuous two-channel inference before comparing controls"
        raise ValueError(message)
    trained = json.loads((Path(index["model_study"]) / "training/result.json").read_text())
    sources = {**trained["identity"]["source_hashes"]}
    sources.update(json.loads((study / "inference_sources.json").read_text()))
    for name, expected in sources.items():
        if sha256_file(name) != expected:
            message = f"Audited model/inference source changed: {name}"
            raise ValueError(message)
    names = (
        "src/ssunet/confocal_motion_comparison.py",
        "src/tools/confocal_motion_compare.py",
        "src/tools/confocal_motion_comparison.html",
        "src/tools/confocal_motion_report.py",
        "src/tools/confocal_motion_review.py",
        "src/ssunet/confocal_motion.py",
    )
    write_json(output / "protocol.json", {**config, "sources": {n: sha256_file(n) for n in names}})
    return index


def _inputs(study, original, clip):
    record = original["recordings"][clip["recording"]]
    names = list(original["recordings"])
    regions = record["regions"]
    roi_index = next(i for i, item in enumerate(regions) if item["name"] == clip["phase"])
    pool_path = record.get(
        "cached_pool", study / "cache" / clip["recording"] / "training_counts.npy"
    )
    pool = np.load(pool_path, mmap_mode="r", allow_pickle=False)
    selected = select_region(pool, record, regions[roi_index], original)
    seed = original["split_seed"] + 1000 * names.index(clip["recording"]) + 10 * roi_index
    a, b = split_motion_pool(
        selected, seed=seed, first_time=original["core_time_start"] - original["temporal_halo"]
    )
    return a, b


def _analyze_clip(study, output, original, options, clip):
    folder = study / "examples" / clip["directory"]
    observed = np.load(folder / "observed.npy", mmap_mode="r", allow_pickle=False)
    reserved = np.load(folder / "heldout.npy", mmap_mode="r", allow_pickle=False)
    network = np.load(folder / "gap4d_probability.npy", mmap_mode="r", allow_pickle=False)
    a, b = _inputs(study, original, clip)
    halo, frames = original["temporal_halo"], original["core_frames"]
    np.testing.assert_array_equal(
        a[:, halo : halo + frames] + b[:, halo : halo + frames], observed - reserved
    )
    if np.any(reserved > observed):
        message = "Reserved photons must be a subset of the observations"
        raise ValueError(message)
    shape = (observed.shape[0], frames, *observed.shape[-2:])
    projections = {key: np.empty(shape, dtype=np.float32) for key in DISPLAY if key != "observed"}
    projections.update(
        observed=observed.mean(axis=2, dtype=np.float32),
        reserved=reserved.mean(axis=2, dtype=np.float32),
    )
    scores = {key: [] for key in METHODS}
    for channel in range(observed.shape[0]):
        LOGGER.info("%s | %s", clip["directory"], clip["channel_names"][channel])
        counts = reserved[channel]
        total = int(observed[channel].sum(dtype=np.int64))
        for method, temporal in (
            ("gaussian3d", 0),
            ("gaussian4d", options["gaussian_temporal_sigma"]),
        ):
            q = gaussian_motion_probability(
                a[channel],
                b[channel],
                sigma=(temporal, *options["gaussian_spatial_sigma_zyx"]),
                halo=halo,
                frames=frames,
                uniform_mixture=options["uniform_mixture"],
            )
            scores[method].append(photon_log_score(q, counts))
            projections[method][channel] = q.mean(axis=1, dtype=np.float64) * total
            del q
        q = network[channel]
        scores["gap4d"].append(photon_log_score(q, counts))
        projections["gap4d"][channel] = q.mean(axis=1, dtype=np.float64) * total
        frozen = factorized_motion_probability(q)
        scores["gap4d_frozen"].append(photon_log_score(frozen, counts))
        del frozen
        photons = int(counts.sum(dtype=np.int64))
        loss = math.log(counts.size)
        scores["uniform"].append(
            {"photons": photons, "negative_log_likelihood": photons * loss, "nats_per_photon": loss}
        )
    trajectories = {
        name: block_trajectories(
            values, clip["voxel_spacing_zyx_um"], block=options["registration_block_frames"]
        )
        for name, values in projections.items()
    }
    destination = output / "clips" / clip["directory"]
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "projections.npz", **projections)
    checksums = {
        name: sha256_file(folder / name)
        for name in ("observed.npy", "heldout.npy", "gap4d_probability.npy")
    }
    row = {**clip, "scores": scores, "trajectories": trajectories, "input_sha256": checksums}
    write_json(destination / "comparison.json", row)
    return row


def _pooled(rows):
    result = {}
    for recording in dict.fromkeys(row["recording"] for row in rows):
        selected = [row for row in rows if row["recording"] == recording]
        result[recording] = {}
        for method in METHODS:
            per_channel = []
            for channel in range(2):
                photons = sum(row["scores"][method][channel]["photons"] for row in selected)
                loss = math.fsum(
                    row["scores"][method][channel]["negative_log_likelihood"] for row in selected
                )
                per_channel.append(
                    {
                        "photons": photons,
                        "negative_log_likelihood": loss,
                        "nats_per_photon": loss / photons if photons else None,
                    }
                )
            photons = sum(item["photons"] for item in per_channel)
            loss = math.fsum(item["negative_log_likelihood"] for item in per_channel)
            result[recording][method] = {
                "channels": per_channel,
                "photons": photons,
                "nats_per_photon": loss / photons if photons else None,
            }
    return result


def _movie(output, row, projections):
    maximum = [
        max(float(np.quantile(np.stack([projections[m][c] for m in DISPLAY]), 0.997)), 0.01)
        for c in range(2)
    ]
    _, times, y, x = projections["observed"].shape
    palette, frames = _fixed_palette(), []
    for time in range(times):
        frame = Image.new("RGB", (2 * x + 16, 2 * (y + 24) + 42), "black")
        draw = ImageDraw.Draw(frame)
        draw.text(
            (2, 0),
            f"T={row['source_start_tzyx'][0] + time} | "
            f"+{time * row['volume_interval_seconds']:.1f}s",
            fill="white",
        )
        draw.text((2, 14), "HyD2 blue; HyD4 green; Z mean", fill="white")
        for position, method in enumerate(DISPLAY):
            rgb = np.zeros((y, x, 3), dtype=np.float32)
            rgb[..., 2] = projections[method][0, time] / maximum[0]
            rgb[..., 1] = projections[method][1, time] / maximum[1]
            left, top = (position % 2) * (x + 16), 38 + (position // 2) * (y + 24)
            draw.text((left, top), METHODS.get(method, "Observed, noisy"), fill="white")
            frame.paste(
                Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)), (left, top + 18)
            )
        frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
    relative = f"clips/{row['directory']}/comparison.gif"
    frames[0].save(
        output / relative,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
        optimize=False,
    )
    return {"path": relative, "maximum_counts_per_voxel": maximum, "display_fps": 5}


def _plots(output, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 2.8 * len(rows)), layout="constrained")
    keys = ("gaussian3d", "gaussian4d", "gap4d", "gap4d_frozen")
    for position, row in enumerate(rows):
        for channel, color in enumerate(("blue", "green")):
            values = [row["scores"][key][channel]["nats_per_photon"] for key in keys]
            axis = axes[position, channel]
            axis.barh(np.arange(4), values, color=["#999999", "#666666", color, "#cccccc"])
            axis.set_yticks(np.arange(4), [METHODS[key] for key in keys], fontsize=8)
            axis.invert_yaxis()
            axis.set_xlim(
                min(values) - 0.05,
                max(max(values), row["scores"]["uniform"][channel]["nats_per_photon"]) + 0.03,
            )
            axis.axvline(
                row["scores"]["uniform"][channel]["nats_per_photon"],
                color="black",
                linestyle=":",
                label="Uniform",
            )
            axis.set_title(f"{row['directory']} | {row['channel_names'][channel]}", fontsize=10)
            axis.set_xlabel("Reserved-photon NLL (nats/photon); lower is better")
            for i, value in enumerate(values):
                axis.text(value + 0.002, i, f"{value:.3f}", va="center", fontsize=8)
    fig.suptitle("128-time arrival distributions | fixed controls | same reserved photons")
    fig.savefig(output / "photon_scores.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 2.8 * len(rows)), layout="constrained")
    styles = {
        "observed": ("#999999", ":"),
        "reserved": ("black", "--"),
        "gaussian3d": ("#c46b00", "-"),
        "gaussian4d": ("#b000b0", "-"),
        "gap4d": (None, "-"),
    }
    for position, row in enumerate(rows):
        for channel, color in enumerate(("blue", "green")):
            axis = axes[position, channel]
            for method, (line_color, style) in styles.items():
                trajectory = row["trajectories"][method][channel]
                x = [v["dx_um"] if v else np.nan for v in trajectory]
                y = [v["dy_um"] if v else np.nan for v in trajectory]
                label = METHODS.get(
                    method, {"observed": "Observed", "reserved": "Reserved photons"}.get(method)
                )
                axis.plot(
                    x, y, style, color=line_color or color, marker=".", linewidth=1.5, label=label
                )
                axis.plot(x[-1], y[-1], "o", color=line_color or color, markersize=4)
            axis.set(
                title=f"{row['directory']} | {row['channel_names'][channel]}",
                xlabel="Apparent X displacement (um)",
                ylabel="Apparent Y displacement (um)",
            )
            axis.grid(alpha=0.2)
            axis.set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Whole-pattern XY registration | 16-time blocks; circles mark final block\n"
        "Not filament tracks or clean ground truth"
    )
    fig.savefig(output / "bulk_motion.png", dpi=150)
    plt.close(fig)


def _html(output, result):
    sections = []
    for row in result["clips"]:
        table = []
        for channel, detector in enumerate(row["channel_names"]):
            for method in METHODS:
                score = row["scores"][method][channel]["nats_per_photon"]
                endpoint = row["trajectories"].get(method, [[], []])[channel]
                displacement = (
                    math.hypot(endpoint[-1]["dx_um"], endpoint[-1]["dy_um"])
                    if endpoint and endpoint[-1]
                    else None
                )
                motion = f"{displacement:.3f}" if displacement is not None else "—"
                table.append(
                    f"<tr><td>{detector}</td><td>{METHODS[method]}</td><td>{score:.4f}</td><td>{motion}</td></tr>"
                )
        sections.append(
            f"<h2>{html.escape(row['recording_name'])} | {row['phase']}</h2>"
            f"<img class='movie' src='{row['movie']['path']}'>"
            "<table><tr><th>Detector</th><th>Prediction</th><th>NLL, nats/photon</th>"
            f"<th>Net bulk displacement, um</th></tr>{''.join(table)}</table>"
        )
    images = "".join(
        "<img class='figure' src='data:image/png;base64,"
        + base64.b64encode((output / name).read_bytes()).decode("ascii")
        + "'>"
        for name in ("photon_scores.png", "bulk_motion.png")
    )
    table = []
    for recording, methods in result["pooled"].items():
        for method, score in methods.items():
            table.append(
                f"<tr><td>{recording}</td><td>{METHODS[method]}</td>"
                f"<td>{score['nats_per_photon']:.4f}</td></tr>"
            )
    pooled = (
        "<table><tr><th>Recording</th><th>Prediction</th>"
        "<th>Pooled NLL, nats/photon</th></tr>" + "".join(table) + "</table>"
    )
    template = Template(Path(__file__).with_name("confocal_motion_comparison.html").read_text())
    page = template.substitute(figures=images, sections="".join(sections), pooled=pooled)
    (output / "index.html").write_text(page, encoding="utf-8")


def main():
    """Run the fixed CPU controls once, then export an inspectable visual comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/confocal_motion_comparison.yml")
    )
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = yaml.safe_load(args.config.read_text())
    study, output = Path(config["study"]), Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        result = json.loads((output / "comparison.json").read_text())
    else:
        index = _check_sources(study, output, config)
        original = json.loads((study / "config.json").read_text())
        rows = []
        for clip in index["clips"]:
            write_json(
                output / "progress.json",
                {"status": "running", "clip": clip["directory"], "complete": len(rows)},
            )
            rows.append(_analyze_clip(study, output, original, config, clip))
        result = {
            "status": "analyzed",
            "clips": rows,
            "pooled": _pooled(rows),
            "checkpoint_sha256": index["checkpoint_sha256"],
            "config": config,
        }
        write_json(output / "comparison.json", result)
    for row in result["clips"]:
        with np.load(
            output / "clips" / row["directory"] / "projections.npz", allow_pickle=False
        ) as archive:
            projections = {name: archive[name] for name in DISPLAY}
        row["movie"] = _movie(output, row, projections)
    _plots(output, result["clips"])
    _html(output, result)
    result["status"] = "complete"
    write_json(output / "comparison.json", result)
    write_json(
        output / "report_sources.json",
        {
            str(path): sha256_file(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("confocal_motion_comparison.html"),
            )
        },
    )
    write_json(output / "progress.json", {"status": "complete", "complete": len(result["clips"])})
    LOGGER.info("Report: %s", output / "index.html")


if __name__ == "__main__":
    main()
