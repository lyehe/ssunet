"""Compare observed and denoised continuous motion without asserting filament identities."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

from ..ssunet.confocal_inference import write_json


def bulk_shift(reference, moving):
    """Estimate apparent XY image displacement after smoothing and brightness removal."""
    a, b = (gaussian_filter(image.astype(np.float64), 1.5) for image in (reference, moving))
    a, b = a - a.mean(), b - b.mean()
    if min(float(np.linalg.norm(a)), float(np.linalg.norm(b))) < 1e-10:
        return None
    shift, error, _ = phase_cross_correlation(a, b, normalization=None, upsample_factor=10)
    return {
        "dx_pixels": -float(shift[1]),
        "dy_pixels": -float(shift[0]),
        "registration_error": float(error),
        "within_quarter_field": bool(np.max(np.abs(shift)) < min(a.shape) / 4),
    }


def block_trajectories(projections, spacing, *, block=16):
    """Register whole projected structures, with no claim of tracking individual filaments."""
    channels, times, y, x = projections.shape
    means = projections.reshape(channels, times // block, block, y, x).mean(axis=2)
    rows = []
    for channel in range(channels):
        trajectory = []
        for image in means[channel]:
            shift = bulk_shift(means[channel, 0], image)
            if shift is not None:
                shift.update(
                    dx_um=shift["dx_pixels"] * spacing[2], dy_um=shift["dy_pixels"] * spacing[1]
                )
            trajectory.append(shift)
        rows.append(trajectory)
    return rows


def _fixed_palette():
    colors = [(0, round(g * 255 / 14), round(b * 255 / 14)) for g in range(15) for b in range(15)]
    colors += [(255, 255, 255), (160, 160, 160)]
    colors += [(0, 0, 0)] * (256 - len(colors))
    palette = Image.new("P", (1, 1))
    palette.putpalette([value for color in colors for value in color])
    return palette


def movie(root, clip, observed, expected):
    """Use one fixed blue/green palette and detector contrast range throughout a movie."""
    maximum = [
        max(float(np.quantile(np.stack([observed[c], expected[c]]), 0.997)), 0.01) for c in range(2)
    ]
    frames, palette = [], _fixed_palette()
    _, times, y, x = observed.shape
    dt = clip["volume_interval_seconds"]
    for time in range(times):
        frame = Image.new("RGB", (x * 2 + 16, y + 66), "black")
        for column, values in enumerate((observed, expected)):
            rgb = np.zeros((y, x, 3), dtype=np.float32)
            rgb[..., 2] = values[0, time] / maximum[0]
            rgb[..., 1] = values[1, time] / maximum[1]
            field = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
            frame.paste(field, (column * (x + 16), 48))
        draw = ImageDraw.Draw(frame)
        draw.text(
            (2, 0), f"T={clip['source_start_tzyx'][0] + time} | +{time * dt:.1f} s", fill="white"
        )
        draw.text((2, 14), "HyD2 blue; HyD4 green | Z mean", fill="white")
        draw.text((2, 32), "Observed, noisy", fill="white")
        draw.text((x + 18, 32), "2-channel GAP", fill="white")
        frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
    path = root / "review" / f"{clip['directory']}.gif"
    path.parent.mkdir(exist_ok=True)
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=200, loop=0, optimize=False
    )
    return {
        "path": str(path.relative_to(root)),
        "display_fps": 5,
        "acquired_seconds_per_frame": dt,
        "maximum_counts_per_voxel": maximum,
    }


def _trajectory_plot(root, rows, plt):
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 3.2 * len(rows)), layout="constrained")
    styles = {"denoised": "-", "observed": ":", "reserved": "--"}
    for row_index, row in enumerate(rows):
        times = np.arange(8) * 16 * row["volume_interval_seconds"]
        for channel, (name, color) in enumerate((("HyD2", "blue"), ("HyD4", "green"))):
            for method, trajectories in row["trajectories"].items():
                for axis_index, key in enumerate(("dx_um", "dy_um")):
                    values = [
                        item[key] if item is not None else np.nan for item in trajectories[channel]
                    ]
                    axes[row_index, axis_index].plot(
                        times,
                        values,
                        styles[method],
                        color=color,
                        label=f"{name}: {method}",
                        linewidth=1.5,
                    )
        for axis_index, direction in enumerate(("X", "Y")):
            axis = axes[row_index, axis_index]
            axis.set_title(f"{row['directory']} | {direction}", fontsize=10)
            axis.set(xlabel="Seconds from first block center", ylabel="Apparent displacement (um)")
            axis.axhline(0, color="gray", linewidth=0.5)
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Whole-pattern XY registration | 16-time blocks | not individual filament tracks")
    fig.savefig(root / "motion_displacements.png", dpi=150)
    plt.close(fig)


def _profile_plot(root, rows, plt):
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 2.8 * len(rows)), layout="constrained")
    for row_index, row in enumerate(rows):
        for direction_index, (direction, spacing_index) in enumerate((("X", 2), ("Y", 1))):
            for column, method in enumerate(("observed", "denoised")):
                profile = row["profiles"][method][direction]
                normalized = profile / np.maximum(profile.sum(-1, keepdims=True), 1e-20)
                axis = axes[row_index, direction_index * 2 + column]
                axis.imshow(
                    normalized,
                    cmap="Greens",
                    aspect="auto",
                    origin="lower",
                    extent=(
                        0,
                        normalized.shape[1] * row["voxel_spacing_zyx_um"][spacing_index],
                        0,
                        normalized.shape[0] * row["volume_interval_seconds"],
                    ),
                )
                axis.set(
                    xlabel=f"{direction} (um)",
                    ylabel="Time (s)",
                    title=f"{row['directory']}\n{method} | {direction}",
                )
    fig.suptitle("HyD4 X/Y spatial profiles; brightness normalized within each time")
    fig.savefig(root / "hyd4_space_time.png", dpi=150)
    plt.close(fig)


def _endpoint_plot(root, rows, plt):
    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 3.1 * len(rows)), layout="constrained")
    for row_index, row in enumerate(rows):
        for method in range(2):
            for endpoint in range(2):
                values = row["endpoints"][method, endpoint]
                rgb = np.zeros((*values.shape[1:], 3), dtype=np.float32)
                rgb[..., 2] = values[0] / row["movie"]["maximum_counts_per_voxel"][0]
                rgb[..., 1] = values[1] / row["movie"]["maximum_counts_per_voxel"][1]
                axis = axes[row_index, method * 2 + endpoint]
                axis.imshow(np.clip(rgb, 0, 1))
                label = "observed" if method == 0 else "denoised"
                edge = "first 16 times" if endpoint == 0 else "last 16 times"
                axis.set_title(f"{row['directory']}\n{label} | {edge}", fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
    fig.suptitle(
        "Endpoint means over 16 times and 16 Z planes | no clean GT\n"
        "HyD2 blue; HyD4 green; matched detector contrast within each clip"
    )
    fig.savefig(root / "motion_endpoints.png", dpi=150)
    plt.close(fig)


def _html(root, rows):
    sections = []
    for row in rows:
        table = []
        for channel, name in enumerate(("HyD2", "HyD4")):
            for method, trajectories in row["trajectories"].items():
                item = trajectories[channel][-1]
                if item is not None:
                    table.append(
                        f"<tr><td>{name}</td><td>{method}</td><td>{item['dx_um']:.3f}</td>"
                        f"<td>{item['dy_um']:.3f}</td><td>{item['registration_error']:.3f}</td></tr>"
                    )
        sections.append(
            f"<h2>{row['recording_name']} | {row['phase']}</h2>"
            f"<p>128 times; {row['duration_seconds']:.1f} seconds; "
            f"{row['movie']['acquired_seconds_per_frame']:.3f} s per acquired volume. "
            "Movie playback is accelerated to 5 fps. Projection is a Z mean.</p>"
            f"<img class='movie' src='{row['movie']['path']}'>"
            "<table><tr><th>Detector</th><th>Data</th><th>Net X (um)</th><th>Net Y (um)</th>"
            "<th>Registration error</th></tr>" + "".join(table) + "</table>"
        )
    figures = "".join(
        '<img class="figure" src="data:image/png;base64,'
        + base64.b64encode((root / name).read_bytes()).decode("ascii")
        + '">'
        for name in ("motion_displacements.png", "hyd4_space_time.png", "motion_endpoints.png")
    )
    page = (
        "<!doctype html><html><meta charset='utf-8'><title>Two-channel motion review</title>"
        "<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:0 18px}"
        ".movie{width:600px;max-width:100%;image-rendering:pixelated}.figure{width:100%}"
        "th,td{padding:5px 12px;border-bottom:1px solid #ddd}</style>"
        "<h1>Microtubule 3 and 4: two-channel 4D review</h1>"
        "<p>The same final two-channel model trained on microtubule 3 is applied to both "
        "recordings; "
        "microtubule 4 is a transfer test. The clips use real 14-time halos and one probability "
        "normalization per detector over the complete continuous clip. Static-input and "
        "overlapping "
        "chunk checks passed. HyD2 is blue; HyD4 is green.</p>"
        "<p>Registration describes projected whole-pattern displacement, not microtubule tracks. "
        "Stage/cell drift, focus and changing brightness can also change these patterns. "
        "The reserved 20% photon observations were excluded from network inputs and training, "
        "and provide an independent "
        "comparison. Registration error is not a confidence interval or biological p-value. "
        "Expected-count maps use each detector's total observed photons for display calibration. "
        "No clean ground truth is available, and fine filament identities are not established.</p>"
        + figures
        + "".join(sections)
        + "</html>"
    )
    (root / "index.html").write_text(page, encoding="utf-8")


def main():
    """Export movies, coarse-displacement measurements and a review of their limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("data/confocal_twochannel_motion_v1"))
    args = parser.parse_args()
    root = args.study
    index = json.loads((root / "index.json").read_text())
    if index["status"] != "complete":
        message = "Finish both recordings before comparing their motion"
        raise ValueError(message)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for clip in index["clips"]:
        folder = root / "examples" / clip["directory"]
        projections = {}
        for method, filename in (
            ("observed", "observed.npy"),
            ("denoised", "gap4d_expected_counts.npy"),
            ("reserved", "heldout.npy"),
        ):
            values = np.load(folder / filename, mmap_mode="r", allow_pickle=False)
            projections[method] = values.mean(axis=2, dtype=np.float32)
        trajectories = {
            method: block_trajectories(values, clip["voxel_spacing_zyx_um"])
            for method, values in projections.items()
        }
        row = {
            **clip,
            "trajectories": trajectories,
            "duration_seconds": (projections["observed"].shape[1] - 1)
            * clip["volume_interval_seconds"],
        }
        row["movie"] = movie(root, clip, projections["observed"], projections["denoised"])
        row["profiles"] = {
            method: {"X": values[1].mean(axis=1), "Y": values[1].mean(axis=2)}
            for method, values in projections.items()
            if method != "reserved"
        }
        row["endpoints"] = np.stack(
            [
                np.stack([values[:, :16].mean(axis=1), values[:, -16:].mean(axis=1)])
                for method, values in projections.items()
                if method != "reserved"
            ]
        )
        rows.append(row)
    _trajectory_plot(root, rows, plt)
    _profile_plot(root, rows, plt)
    _endpoint_plot(root, rows, plt)
    _html(root, rows)
    summary = [
        {key: value for key, value in row.items() if key not in {"profiles", "endpoints"}}
        for row in rows
    ]
    write_json(
        root / "motion_summary.json",
        {
            "status": "complete",
            "clips": summary,
            "scope": "Exploratory bulk XY displacement; no validated individual microtubule tracks",
        },
    )


if __name__ == "__main__":
    main()
