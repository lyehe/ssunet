"""Detect afterpulsing artifacts in SPAD data.

Compares each pixel's run-length statistics against what iid Bernoulli
predicts given the local spatial firing rate. Key diagnostics:
  - P(fire at t+1 | fire at t) vs P(fire at t+1 | no fire at t)
  - Run-length distribution vs expected geometric distribution
  - Per-pixel z-scores for excess consecutive firings

Usage:
    python detect_afterpulsing.py D:/Yehe/PC/reconstruction/test/blue-bathroom
    python detect_afterpulsing.py D:/Yehe/PC/reconstruction/test/blue-bathroom --chunks 10
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter


def compute_local_rates(frames, spatial_radius=7):
    """Compute local spatial firing rate for each pixel (excluding self)."""
    n_frames = frames.shape[0]
    kernel_size = 2 * spatial_radius + 1
    kernel_area = kernel_size * kernel_size
    obs_rate = frames.sum(axis=0).astype(np.float64) / n_frames  # (H, W, C)
    local_rates = np.zeros_like(obs_rate)
    for ch in range(frames.shape[3]):
        local_mean = uniform_filter(obs_rate[:, :, ch], size=kernel_size, mode="reflect")
        local_rates[:, :, ch] = np.clip(
            (local_mean * kernel_area - obs_rate[:, :, ch]) / (kernel_area - 1), 0, 1
        )
    return obs_rate, local_rates


def detect_afterpulsing(frames, local_rates):  # noqa: C901
    """Analyze afterpulsing statistics for one chunk.

    Returns dict with global and per-pixel diagnostics.
    """
    n_frames, height, width, channels = frames.shape
    total_pixels = height * width * channels
    fires = frames.astype(np.float64)

    # --- Global conditional probabilities ---
    total_firings = fires.sum()
    total_rate = total_firings / (n_frames * total_pixels)

    consec = (fires[:-1] * fires[1:]).sum()
    n_fire_frames = fires[:-1].sum()
    n_nofire_frames = (1 - fires[:-1]).sum()
    fire_after_nofire = ((1 - fires[:-1]) * fires[1:]).sum()

    p_fire_after_fire = consec / max(n_fire_frames, 1)
    p_fire_after_nofire = fire_after_nofire / max(n_nofire_frames, 1)

    # --- Per-pixel: compare observed runs vs local spatial rate ---
    per_pixel_consec = (fires[:-1] * fires[1:]).sum(axis=0)

    # Expected consecutive under iid using LOCAL rate (not own rate)
    expected_consec = (n_frames - 1) * local_rates**2
    expected_std = np.sqrt((n_frames - 1) * local_rates**2 * (1 - local_rates**2) + 1e-10)
    z_scores = np.where(
        expected_std > 0,
        (per_pixel_consec - expected_consec) / expected_std,
        0,
    )

    # --- Run-length distribution (full scan) ---
    max_run_global = 0
    max_run_bins = min(n_frames, 128) + 1
    run_counts = np.zeros(max_run_bins, dtype=np.int64)
    # Also track: for runs >= threshold, what is the local rate?
    long_run_info = []  # (row, col, ch, run_len, local_p)

    for ch in range(channels):
        for r in range(height):
            for c in range(width):
                run = 0
                for t in range(n_frames):
                    if frames[t, r, c, ch] == 1:
                        run += 1
                    else:
                        if run > 0:
                            idx = min(run, max_run_bins - 1)
                            run_counts[idx] += 1
                            if run > max_run_global:
                                max_run_global = run
                            p_local = local_rates[r, c, ch]
                            # Flag runs that are implausible: p^(k-1) < 1e-6
                            if p_local > 0 and p_local < 1:
                                log_prob = (run - 1) * np.log(p_local)
                                if log_prob < np.log(1e-6):
                                    long_run_info.append((r, c, ch, run, p_local, log_prob))
                        run = 0
                # Handle run at end of frames
                if run > 0:
                    idx = min(run, max_run_bins - 1)
                    run_counts[idx] += 1
                    if run > max_run_global:
                        max_run_global = run
                    p_local = local_rates[r, c, ch]
                    if p_local > 0 and p_local < 1:
                        log_prob = (run - 1) * np.log(p_local)
                        if log_prob < np.log(1e-6):
                            long_run_info.append((r, c, ch, run, p_local, log_prob))

    return {
        "total_rate": total_rate,
        "p_fire_after_fire": p_fire_after_fire,
        "p_fire_after_nofire": p_fire_after_nofire,
        "afterpulse_excess": p_fire_after_fire - p_fire_after_nofire,
        "afterpulse_ratio": p_fire_after_fire / max(p_fire_after_nofire, 1e-10),
        "n_consec_observed": int(consec),
        "n_consec_expected_iid": float((n_frames - 1) * total_rate**2 * total_pixels),
        "n_pixels_z3": int((z_scores > 3).sum()),
        "n_pixels_z5": int((z_scores > 5).sum()),
        "n_pixels_z10": int((z_scores > 10).sum()),
        "total_pixels": total_pixels,
        "max_run_length": max_run_global,
        "run_counts": run_counts,
        "n_implausible_runs": len(long_run_info),
        "long_run_info": long_run_info,
    }


def main():
    """Detect afterpulsing in the SPAD data named on the command line."""
    parser = argparse.ArgumentParser(description="Detect afterpulsing in SPAD data")
    parser.add_argument("scene_dir", type=str, help="Path to scene directory")
    parser.add_argument(
        "--chunks", type=int, default=3, help="Number of chunks to analyze (default: 3)"
    )
    parser.add_argument(
        "--spatial-radius", type=int, default=7, help="Spatial neighborhood radius (default: 7)"
    )
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    scene_name = scene_dir.name
    npy_files = sorted(scene_dir.glob("*.npy"))
    n_analyze = min(args.chunks, len(npy_files))

    print(f"Scene: {scene_name} ({len(npy_files)} chunks, analyzing {n_analyze})")
    print(f"Spatial radius: {args.spatial_radius}")
    print()

    for i in range(n_analyze):
        npy_path = npy_files[i]
        packed = np.load(str(npy_path))
        unpacked = np.unpackbits(packed, axis=2)[:, :, :800, :]
        n_frames = unpacked.shape[0]

        _obs_rate, local_rates = compute_local_rates(unpacked, args.spatial_radius)
        stats = detect_afterpulsing(unpacked, local_rates)

        print(
            f"--- {npy_path.name} ({n_frames} frames, "
            f"{unpacked.shape[1]}x{unpacked.shape[2]}x{unpacked.shape[3]}) ---"
        )
        print(f"  Global firing rate:       {stats['total_rate']:.4f}")
        print(f"  P(fire | prev fire):      {stats['p_fire_after_fire']:.6f}")
        print(f"  P(fire | prev no-fire):   {stats['p_fire_after_nofire']:.6f}")
        print(f"  Afterpulse excess:        {stats['afterpulse_excess']:.6f}")
        print(f"  Afterpulse ratio:         {stats['afterpulse_ratio']:.4f}x")
        print(
            f"  Consecutive pairs:        {stats['n_consec_observed']:,} "
            f"(expected iid: {stats['n_consec_expected_iid']:,.0f})"
        )
        print(
            f"  Pixels >3sigma (vs local): {stats['n_pixels_z3']:,} / {stats['total_pixels']:,} "
            f"({100 * stats['n_pixels_z3'] / stats['total_pixels']:.2f}%)"
        )
        print(f"  Pixels >5sigma (vs local): {stats['n_pixels_z5']:,}")
        print(f"  Pixels >10sigma:           {stats['n_pixels_z10']:,}")
        print(f"  Max run length:            {stats['max_run_length']}")

        # Run-length distribution
        rc = stats["run_counts"]
        print("  Run-length distribution:")
        for k in range(1, min(20, len(rc))):
            if rc[k] > 0:
                print(f"    len={k:3d}: {int(rc[k]):>10,}")

        # Implausible runs summary
        n_imp = stats["n_implausible_runs"]
        print(f"  Implausible runs (p^(k-1) < 1e-6): {n_imp:,}")
        if n_imp > 0:
            # Show a few examples
            examples = stats["long_run_info"][:10]
            for r, c, ch, run_len, p_local, log_prob in examples:
                print(
                    f"    pixel ({r},{c},ch{ch}): run={run_len}, "
                    f"local_p={p_local:.4f}, log_prob={log_prob:.1f}"
                )
            if n_imp > 10:
                print(f"    ... and {n_imp - 10} more")

        print()


if __name__ == "__main__":
    main()
