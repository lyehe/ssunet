"""Make recorded photons visible as markers without changing detection decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def recorded_hit_figure(root):
    """Plot the same deterministic examples as the frozen report, including empty A."""
    config = json.loads((root / "config.json").read_text())
    with np.load(root / "test_observations.npz", allow_pickle=False) as handle:
        offsets, locations, retained = (
            handle[key].copy() for key in ("offsets", "locations", "retained")
        )
    with np.load(root / "evaluation_only_truth.npz", allow_pickle=False) as handle:
        strength = handle["expected_signal"].copy()
    with np.load(root / "frozen_reference.npz", allow_pickle=False) as handle:
        edges = handle["edges"].copy()
        sensors, bins = handle["prior"].shape
    levels = config["generator"]["signal_expected_photons"]
    selected = (levels[1], config["evaluation"]["primary_expected_signal"], levels[-2])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained", sharey=True)
    centers = (edges[:-1] + edges[1:]) / 2000
    for axis, level in zip(axes, selected, strict=True):
        event = int(np.flatnonzero(strength == level)[0])
        rows = slice(offsets[event], offsets[event + 1])
        photons, half = locations[rows], retained[rows]
        for mask, label, color, marker in (
            (half, "Recorded A", "#2676b6", "o"),
            (~half, "Recorded B", "#c76620", "x"),
        ):
            cells, counts = np.unique(photons[mask], return_counts=True)
            sensor, time = np.divmod(cells, bins)
            axis.scatter(
                centers[time],
                sensor,
                s=30 + 15 * counts,
                color=color,
                marker=marker,
                label=f"{label}: {int(mask.sum())} photons",
            )
        axis.set_xlim(edges[0] / 1000, edges[-1] / 1000)
        axis.set_ylim(-0.5, sensors - 0.5)
        axis.set_xlabel("Fixed-window time (microseconds)")
        axis.set_title(f"Expected signal {level:g}; window ID {event}")
        axis.legend(fontsize=8, loc="lower right")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Sensor index")
    figure.suptitle("Recorded photons in the deterministic examples: larger markers show arrivals")
    figure.supxlabel(
        "A/B colors indicate random splitting of recorded photons, not signal/background truth. "
        "Marker area increases with bin multiplicity.",
        fontsize=9,
    )
    for extension in ("png", "svg"):
        figure.savefig(root / f"recorded_hits.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    """Render cached observations without fitting, recalibrating or reopening selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("data/neutrino/detection_study_v1"))
    args = parser.parse_args()
    recorded_hit_figure(args.study)


if __name__ == "__main__":
    main()
