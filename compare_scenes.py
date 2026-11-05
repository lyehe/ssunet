"""Compare reconstruction output quality across scenes.

Analyzes output PNGs for noise/quality indicators and compares
input data statistics between blue-bathroom (problematic) and other scenes.
"""

from pathlib import Path

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import laplace

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


OUTPUT_DIR = Path("C:/Users/jenkinslab/project/ssunet/output")
TEST_DIR = Path("D:/Yehe/PC/reconstruction/test")

# Scenes to analyze (main output dirs that have PNGs)
SCENE_DIRS = [
    "attic",
    "basement",
    "bedroom-interior",
    "blue-bathroom",
    "coffee-office-bar",
    "train-passby",
]

# Frames to sample
SAMPLE_FRAMES = [0, 10, 20, 36]


def compute_laplacian_energy(img_gray: np.ndarray) -> float:
    """Compute high-frequency energy via Laplacian filter magnitude."""
    if HAS_SCIPY:
        lap = laplace(img_gray.astype(np.float64))
        return float(np.mean(np.abs(lap)))
    else:
        # Manual 2D Laplacian kernel convolution
        k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        g = img_gray.astype(np.float64)
        # Simple convolution via correlation
        h, w = g.shape
        out = np.zeros((h - 2, w - 2), dtype=np.float64)
        for di in range(3):
            for dj in range(3):
                out += k[di, dj] * g[di : di + h - 2, dj : dj + w - 2]
        return float(np.mean(np.abs(out)))


def analyze_png(filepath: Path) -> dict:
    """Compute per-image statistics for a PNG."""
    img = np.array(Image.open(filepath))
    gray = np.mean(img.astype(np.float64), axis=2) if img.ndim == 3 else img.astype(np.float64)

    stats = {
        "shape": img.shape,
        "dtype": str(img.dtype),
        "min": int(img.min()),
        "max": int(img.max()),
        "mean": float(img.mean()),
        "std": float(img.std()),
        "laplacian_energy": compute_laplacian_energy(gray),
    }

    # Per-channel stats if RGB
    if img.ndim == 3 and img.shape[2] == 3:
        for c, name in enumerate(["R", "G", "B"]):
            ch = img[:, :, c].astype(np.float64)
            stats[f"{name}_mean"] = float(ch.mean())
            stats[f"{name}_std"] = float(ch.std())

    return stats


def analyze_scene_outputs():
    """Analyze output PNGs for all scenes."""
    print("=" * 80)
    print("PART 1: OUTPUT PNG ANALYSIS")
    print("=" * 80)

    scene_summaries = {}

    for scene in SCENE_DIRS:
        scene_dir = OUTPUT_DIR / scene
        if not scene_dir.exists():
            print(f"\n--- {scene}: directory not found ---")
            continue

        pngs = sorted(scene_dir.glob("*.png"))
        if not pngs:
            print(f"\n--- {scene}: no PNGs found ---")
            continue

        print(f"\n{'=' * 60}")
        print(f"Scene: {scene}  ({len(pngs)} PNGs)")
        print(f"{'=' * 60}")

        frame_stats = []
        for frame_idx in SAMPLE_FRAMES:
            fname = f"{frame_idx:06d}.png"
            fpath = scene_dir / fname
            if not fpath.exists():
                continue

            stats = analyze_png(fpath)
            frame_stats.append(stats)

            print(
                f"\n  Frame {frame_idx:3d}: mean={stats['mean']:7.2f}  std={stats['std']:6.2f}  "
                f"min={stats['min']:3d}  max={stats['max']:3d}  "
                f"laplacian={stats['laplacian_energy']:6.2f}"
            )
            if "R_mean" in stats:
                print(
                    f"             R={stats['R_mean']:6.2f}({stats['R_std']:5.2f})  "
                    f"G={stats['G_mean']:6.2f}({stats['G_std']:5.2f})  "
                    f"B={stats['B_mean']:6.2f}({stats['B_std']:5.2f})"
                )

        if frame_stats:
            avg_mean = np.mean([s["mean"] for s in frame_stats])
            avg_std = np.mean([s["std"] for s in frame_stats])
            avg_lap = np.mean([s["laplacian_energy"] for s in frame_stats])
            scene_summaries[scene] = {
                "avg_mean": avg_mean,
                "avg_std": avg_std,
                "avg_laplacian": avg_lap,
                "n_frames": len(pngs),
            }

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SCENE SUMMARY COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'Scene':<25} {'Avg Mean':>10} {'Avg Std':>10} {'Avg Laplacian':>15} {'Frames':>8}")
    print("-" * 70)
    for scene, s in sorted(scene_summaries.items()):
        marker = " *** PROBLEMATIC" if scene == "blue-bathroom" else ""
        print(
            f"{scene:<25} {s['avg_mean']:10.2f} {s['avg_std']:10.2f} "
            f"{s['avg_laplacian']:15.2f} {s['n_frames']:8d}{marker}"
        )


def analyze_input_data():
    """Compare input data statistics between blue-bathroom and other scenes."""
    print(f"\n\n{'=' * 80}")
    print("PART 2: INPUT DATA STATISTICS (test .npy files)")
    print("=" * 80)

    scenes_to_compare = ["blue-bathroom", "coffee-office-bar", "basement", "bedroom-interior"]

    for scene in scenes_to_compare:
        scene_dir = TEST_DIR / scene
        if not scene_dir.exists():
            print(f"\n--- {scene}: test directory not found ---")
            continue

        npys = sorted(scene_dir.glob("*.npy"))
        print(f"\n{'=' * 60}")
        print(f"Scene: {scene}  ({len(npys)} npy files)")
        print(f"{'=' * 60}")

        # Load first file for detailed analysis
        if npys:
            data = np.load(str(npys[0]))
            print(f"  File: {npys[0].name}")
            print(f"  Shape: {data.shape}  Dtype: {data.dtype}")
            print(f"  Min: {data.min():.4f}  Max: {data.max():.4f}")
            print(f"  Mean: {data.mean():.6f}  Std: {data.std():.6f}")

            # Bit density (fraction of 1s for binary data)
            if data.dtype == np.uint8:
                bit_density = data.mean() / 255.0
                print(f"  Bit density (mean/255): {bit_density:.6f}")

            # Per-channel stats (last dim is channels=3)
            if data.ndim == 4 and data.shape[3] == 3:
                h, w, t, c = data.shape
                print(f"  Spatial: {h}x{w}, Temporal: {t} frames, Channels: {c}")
                for ch_idx, ch_name in enumerate(["R", "G", "B"]):
                    ch_data = data[:, :, :, ch_idx].astype(np.float64)
                    print(
                        f"    {ch_name}: mean={ch_data.mean():.6f}  std={ch_data.std():.6f}  "
                        f"nonzero_frac={np.count_nonzero(ch_data) / ch_data.size:.6f}"
                    )

                # Frame-by-frame mean (temporal variation)
                frame_means = [data[:, :, t_i, :].mean() for t_i in range(min(t, 10))]
                print(f"  First 10 frame means: {[f'{m:.4f}' for m in frame_means]}")
                all_frame_means = [data[:, :, t_i, :].mean() for t_i in range(t)]
                print(
                    f"  Frame mean range: [{min(all_frame_means):.4f}, {max(all_frame_means):.4f}]"
                )
                print(f"  Frame mean variance: {np.var(all_frame_means):.6f}")

            # Also check a few more npy files for consistency
            if len(npys) > 1:
                file_means = []
                for nf in npys[:5]:
                    d = np.load(str(nf))
                    file_means.append(d.mean())
                print(f"  First {len(file_means)} file means: {[f'{m:.4f}' for m in file_means]}")


def compare_output_npys():
    """Compare output .npy files (model raw output) between scenes."""
    print(f"\n\n{'=' * 80}")
    print("PART 3: OUTPUT NPY (RAW MODEL OUTPUT) COMPARISON")
    print("=" * 80)

    for scene in SCENE_DIRS:
        scene_dir = OUTPUT_DIR / scene
        npys = sorted(scene_dir.glob("*.npy")) if scene_dir.exists() else []
        if not npys:
            continue

        # Load first npy
        data = np.load(str(npys[0]))
        print(f"\n  {scene}:")
        print(f"    Shape: {data.shape}  Dtype: {data.dtype}")
        print(f"    Min: {data.min():.6f}  Max: {data.max():.6f}")
        print(f"    Mean: {data.mean():.6f}  Std: {data.std():.6f}")

        if data.ndim == 3 and data.shape[2] == 3:
            for ch_idx, ch_name in enumerate(["R", "G", "B"]):
                ch = data[:, :, ch_idx]
                print(
                    f"    {ch_name}: mean={ch.mean():.6f}  std={ch.std():.6f}  "
                    f"min={ch.min():.6f}  max={ch.max():.6f}"
                )
        elif data.ndim == 2:
            print("    (grayscale)")

        # Check for negative values or NaN
        if np.any(np.isnan(data)):
            print("    WARNING: contains NaN values!")
        if np.any(np.isinf(data)):
            print("    WARNING: contains Inf values!")
        if data.min() < 0:
            print(f"    NOTE: contains negative values (min={data.min():.6f})")


def cross_scene_laplacian_profile():
    """Compare Laplacian energy across all frames for each scene."""
    print(f"\n\n{'=' * 80}")
    print("PART 4: LAPLACIAN ENERGY ACROSS ALL FRAMES")
    print("=" * 80)

    for scene in SCENE_DIRS:
        scene_dir = OUTPUT_DIR / scene
        if not scene_dir.exists():
            continue
        pngs = sorted(scene_dir.glob("*.png"))
        if not pngs:
            continue

        energies = []
        for p in pngs:
            img = np.array(Image.open(p))
            gray = (
                np.mean(img.astype(np.float64), axis=2) if img.ndim == 3 else img.astype(np.float64)
            )
            energies.append(compute_laplacian_energy(gray))

        print(f"\n  {scene}: ({len(energies)} frames)")
        print(
            f"    Laplacian energy: mean={np.mean(energies):.2f}  std={np.std(energies):.2f}  "
            f"min={np.min(energies):.2f}  max={np.max(energies):.2f}"
        )
        # Show per-frame if few enough
        if len(energies) <= 40:
            # Show in compact form
            vals = " ".join(f"{e:.1f}" for e in energies)
            print(f"    Per-frame: [{vals}]")


if __name__ == "__main__":
    print("Scene Reconstruction Quality Comparison")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Test data dir: {TEST_DIR}")
    print()

    analyze_scene_outputs()
    analyze_input_data()
    compare_output_npys()
    cross_scene_laplacian_profile()

    print(f"\n\n{'=' * 80}")
    print("DONE")
    print("=" * 80)
