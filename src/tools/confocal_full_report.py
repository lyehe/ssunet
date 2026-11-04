"""Audit complete whole-acquisition probabilities and export scientific figures."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ..ssunet.confocal_inference import describe_source, sha256_file, write_json
from .confocal_evaluate import _validate_external_package, _validate_sources


def _check_arrays(root, shape):
    arrays = [
        np.load(root / name, mmap_mode="r", allow_pickle=False)
        for name in ("counts.npy", "probability.npy", "expected_counts.npy")
    ]
    if any(array.shape != tuple(shape) for array in arrays):
        message = "Whole-acquisition output shapes differ from the source"
        raise ValueError(message)
    if [array.dtype for array in arrays] != [np.uint8, np.float32, np.float32]:
        message = "Count/probability output dtypes differ from their documented units"
        raise ValueError(message)
    return arrays


def _figures(root, arrays, metadata, observed_totals, expected_totals):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts, _, expected = arrays
    indices = [0, counts.shape[0] // 2, counts.shape[0] - 1]
    z = counts.shape[1] // 2
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), layout="constrained")
    for row, t in enumerate(indices):
        fields = [counts[t, z], expected[t, z], counts[t].mean(0), expected[t].mean(0)]
        titles = [
            "Observed C | central Z",
            "4D SSL | central Z",
            "Observed C | Z mean (noisy)",
            "4D SSL | Z mean",
        ]
        for col, (field, title) in enumerate(zip(fields, titles, strict=True)):
            vmax = 3 if col == 0 else max(float(np.quantile(field, 0.997)), 0.1)
            artist = axes[row, col].imshow(
                field, vmin=0, vmax=vmax, cmap="magma", interpolation="nearest"
            )
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            fig.colorbar(
                artist, ax=axes[row, col], fraction=0.046, pad=0.02, label="photons / voxel"
            )
        seconds = t * metadata["volume_interval_seconds"]
        axes[row, 0].set_ylabel(f"T={t}\n{seconds:.1f} s", fontsize=10)
    fig.suptitle(
        "Full-acquisition SSL reconstruction | raw counts are noisy observations", fontsize=13
    )
    fig.savefig(root / "whole_acquisition_examples.png", dpi=160)
    fig.savefig(root / "whole_acquisition_examples.svg")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 4), layout="constrained")
    time_seconds = np.arange(len(observed_totals)) * metadata["volume_interval_seconds"]
    axis.plot(time_seconds, observed_totals, label="Observed C", lw=1)
    axis.plot(time_seconds, expected_totals, label="4D expected counts", lw=1)
    axis.set(
        xlabel="Acquisition time (seconds)",
        ylabel="Photons per volume",
        title="Photon-mass time course | count scale estimated from observations",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    fig.savefig(root / "photon_time_course.png", dpi=160)
    plt.close(fig)


def audit(root, study):
    """Check every voxel, global normalization, count calibration and saved identities."""
    progress = json.loads((root / "progress.json").read_text())
    if progress["status"] != "complete":
        message = "Wait until the whole acquisition has finished inference"
        raise RuntimeError(message)
    metadata = json.loads((root / "manifest.json").read_text())
    result = json.loads((root / "result.json").read_text())
    cache = json.loads((root / "counts_manifest.json").read_text())
    _, current_source = describe_source(metadata["source"])
    checkpoint = sha256_file(study / "training/pilot/conv4d/last.pt")
    if (
        current_source["identity"] != metadata["identity"]
        or checkpoint != metadata["checkpoint_sha256"]
        or checkpoint != result["checkpoint_sha256"]
        or result["status"] != "passed"
    ):
        message = "Source acquisition or model checkpoint identity changed"
        raise ValueError(message)
    _validate_sources(study, json.loads((study / "source_snapshot.json").read_text()))
    _validate_external_package(study)
    for name, digest in metadata["inference_sources"].items():
        if (
            sha256_file(Path(name)) != digest
            or sha256_file(root / "source_snapshot" / name) != digest
        ):
            message = "Inference source identity changed"
            raise ValueError(message)
    if (
        sha256_file(root / "counts.npy") != cache["sha256"]
        or sha256_file(root / "probability.npy") != result["probability_sha256"]
    ):
        message = "Saved observation or probability data changed"
        raise ValueError(message)
    arrays = _check_arrays(root, metadata["shape_tzyx"])
    counts, q, expected = arrays
    observed_totals, expected_totals = [], []
    mass, maximum_error = 0.0, 0.0
    for t in range(counts.shape[0]):
        if (
            not np.isfinite(q[t]).all()
            or not np.isfinite(expected[t]).all()
            or np.any(q[t] < 0)
            or np.any(expected[t] < 0)
        ):
            message = "Invalid whole-acquisition probability or expected count"
            raise ValueError(message)
        mass += float(q[t].sum(dtype=np.float64))
        observed_totals.append(int(counts[t].sum(dtype=np.uint64)))
        expected_totals.append(float(expected[t].sum(dtype=np.float64)))
        calibrated = (q[t] * result["observed_photons"]).astype(np.float32)
        maximum_error = max(maximum_error, float(np.max(np.abs(expected[t] - calibrated))))
    photon_total = sum(observed_totals)
    relative_error = abs(sum(expected_totals) - photon_total) / max(photon_total, 1)
    if (
        abs(mass - 1) > 1e-6
        or maximum_error > 1e-6
        or relative_error > 1e-6
        or photon_total != result["observed_photons"]
    ):
        message = "Global probability normalization or observed count calibration failed"
        raise ValueError(message)
    report = {
        "status": "passed",
        "shape_tzyx": metadata["shape_tzyx"],
        "probability_sum_float64": mass,
        "observed_photons": photon_total,
        "count_calibration_max_abs_error": maximum_error,
        "count_calibration_relative_mass_error": relative_error,
        "observed_photons_per_volume": observed_totals,
        "expected_photons_per_volume": expected_totals,
        "scope": "SSL reconstruction; no clean-ground-truth or detection accuracy claim",
    }
    write_json(root / "whole_acquisition_audit.json", report)
    _figures(root, arrays, metadata, observed_totals, expected_totals)
    return report


def main():
    """Run the CPU-only completed reconstruction audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, default=Path("data/confocal_full_series013_v1"))
    parser.add_argument(
        "--study", type=Path, default=Path("data/confocal_gap4d_features32_batch4_v1")
    )
    args = parser.parse_args()
    result = audit(args.inference, args.study)
    logging.basicConfig(level=logging.INFO)
    logging.info("Audit passed: probability sum %.10f", result["probability_sum_float64"])


if __name__ == "__main__":
    main()
