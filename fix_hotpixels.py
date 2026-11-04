"""Detect and correct statistically unrealistic pixels in SPAD data.

Two correction stages:
1. **Afterpulsing correction** (temporal, runs first): Detects implausibly long
   runs of consecutive firings by comparing run length against the local spatial
   firing rate. For each flagged run, keeps the first pulse (real photon) and
   resamples the rest with binomial(1, p_local) to match neighborhood statistics.
2. **Hot pixel correction** (spatial): For each 32-frame temporal bin, compares
   each pixel's firing count against its local spatial neighborhood. Pixels
   firing significantly above the local expected rate have excess firings
   randomly removed (binomial thinning).

Afterpulsing runs first so that the hot pixel stage sees clean temporal data.

Output is written to {scene_name}_corrected/ under the same parent directory.

Usage:
    python fix_hotpixels.py D:/Yehe/PC/reconstruction/test/blue-bathroom
    python fix_hotpixels.py D:/Yehe/PC/reconstruction/test/train-passby
    python fix_hotpixels.py D:/Yehe/PC/reconstruction/test/blue-bathroom --dry-run
    python fix_hotpixels.py D:/Yehe/PC/reconstruction/test/blue-bathroom --no-hotpixel
    python fix_hotpixels.py D:/Yehe/PC/reconstruction/test/blue-bathroom --no-afterpulse
    python fix_hotpixels.py D:/Yehe/PC/reconstruction/test/blue-bathroom --ap-threshold 0.001
"""

import argparse
import shutil
from pathlib import Path

import numba
import numpy as np
from scipy.ndimage import uniform_filter


def compute_local_rates(frames, spatial_radius=7):
    """Compute local spatial firing rate for each pixel, excluding self."""
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
    return local_rates


@numba.njit(cache=False)
def thin_outliers(frames, outlier_rows, outlier_cols, outlier_chs, target_counts, rng_seed):
    """Remove excess firings from outlier pixels (numba-accelerated)."""
    n_frames = frames.shape[0]
    np.random.seed(rng_seed)

    for idx in range(len(outlier_rows)):
        r = outlier_rows[idx]
        c = outlier_cols[idx]
        ch = outlier_chs[idx]
        target = target_counts[idx]

        firing = np.empty(n_frames, dtype=numba.int32)
        n_firing = 0
        for t in range(n_frames):
            if frames[t, r, c, ch] == 1:
                firing[n_firing] = t
                n_firing += 1

        if n_firing <= target:
            continue

        for i in range(n_firing):
            frames[firing[i], r, c, ch] = 0

        keep = min(target, n_firing)
        for i in range(keep):
            j = i + np.random.randint(0, n_firing - i)
            firing[i], firing[j] = firing[j], firing[i]
            frames[firing[i], r, c, ch] = 1


@numba.njit(cache=False)
def correct_afterpulsing(frames, local_rates, run_prob_threshold, rng_seed):  # noqa: C901
    """Correct afterpulsing by detecting implausible consecutive-firing runs.

    For each pixel with local spatial rate p, finds runs of consecutive 1s.
    A run of length k has probability p^(k-1) under iid Bernoulli (given
    the first fire). If p^(k-1) < run_prob_threshold, the run is flagged:
      - First frame (real photon) is always kept
      - Each subsequent frame in the run is resampled as binomial(1, p)
        to match the local spatial statistics

    Runs consistent with iid (short runs, bright regions) are untouched.

    Returns (n_runs_corrected, n_firings_removed).
    """
    n_frames, height, width, channels = frames.shape
    np.random.seed(rng_seed)
    total_runs = 0
    total_removed = 0

    for ch in range(channels):
        for r in range(height):
            for c in range(width):
                p = local_rates[r, c, ch]
                if p <= 0.0 or p >= 1.0:
                    continue

                # Minimum run length that triggers correction:
                # p^(k-1) < threshold  =>  k-1 > log(threshold)/log(p)
                log_p = np.log(p)
                log_thresh = np.log(run_prob_threshold)
                min_bad_run = int(log_thresh / log_p) + 2  # +2: ceil + 1 for the first pulse

                if min_bad_run < 2:
                    min_bad_run = 2

                # Scan for runs of consecutive 1s
                run_start = -1
                run_len = 0
                for t in range(n_frames + 1):
                    is_fire = False
                    if t < n_frames:
                        is_fire = frames[t, r, c, ch] == 1
                    if is_fire:
                        if run_len == 0:
                            run_start = t
                        run_len += 1
                    else:
                        if run_len >= min_bad_run:
                            # Implausible run detected
                            total_runs += 1
                            # Keep first frame (real photon)
                            # Resample rest with binomial(1, p)
                            for i in range(run_start + 1, run_start + run_len):
                                if np.random.random() < p:
                                    # Keep (matches local rate)
                                    pass
                                else:
                                    frames[i, r, c, ch] = 0
                                    total_removed += 1
                        run_len = 0

    return total_runs, total_removed


def correct_bin(
    frames: np.ndarray,
    spatial_radius: int = 7,
    sigma_threshold: float = 5.0,
    rng_seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Correct spatial hot pixels in a single temporal bin."""
    n_frames = frames.shape[0]
    kernel_size = 2 * spatial_radius + 1
    kernel_area = kernel_size * kernel_size

    obs_count = frames.sum(axis=0).astype(np.float64)
    obs_rate = obs_count / n_frames

    all_rows = []
    all_cols = []
    all_chs = []
    all_targets = []

    for ch in range(frames.shape[3]):
        rate_ch = obs_rate[:, :, ch]
        local_mean = uniform_filter(rate_ch, size=kernel_size, mode="reflect")
        neighbor_mean = np.clip((local_mean * kernel_area - rate_ch) / (kernel_area - 1), 0, 1)

        expected_count = neighbor_mean * n_frames
        expected_std = np.sqrt(n_frames * neighbor_mean * (1 - neighbor_mean) + 1e-10)

        excess = obs_count[:, :, ch] - expected_count
        is_outlier = (excess > sigma_threshold * expected_std) & (rate_ch > neighbor_mean)

        rows, cols = np.where(is_outlier)
        if len(rows) == 0:
            continue

        targets = np.round(neighbor_mean[rows, cols] * n_frames).astype(np.int32)
        all_rows.append(rows)
        all_cols.append(cols)
        all_chs.append(np.full(len(rows), ch, dtype=np.int32))
        all_targets.append(targets)

    if not all_rows:
        return frames, 0

    outlier_rows = np.concatenate(all_rows).astype(np.int32)
    outlier_cols = np.concatenate(all_cols).astype(np.int32)
    outlier_chs = np.concatenate(all_chs).astype(np.int32)
    target_counts = np.concatenate(all_targets).astype(np.int32)

    n_corrected = len(outlier_rows)
    thin_outliers(frames, outlier_rows, outlier_cols, outlier_chs, target_counts, rng_seed)

    return frames, n_corrected


def fix_chunk(
    packed: np.ndarray,
    bin_size: int = 32,
    spatial_radius: int = 7,
    sigma_threshold: float = 5.0,
    ap_threshold: float = 1e-6,
    base_seed: int = 0,
    do_hotpixel: bool = True,
    do_afterpulse: bool = True,
) -> tuple[np.ndarray, int, int, int]:
    """Fix unrealistic pixels in a single chunk.

    Returns (fixed_packed, n_hotpixel_corrected, n_ap_runs, n_ap_removed).
    """
    unpacked = np.unpackbits(packed, axis=2)[:, :, :800, :]
    n_frames = unpacked.shape[0]
    total_hp = 0
    n_ap_runs = 0
    n_ap_removed = 0

    # Stage 1: Afterpulsing correction FIRST (temporal)
    # Run before hot pixel so spatial stats aren't polluted by stuck runs
    if do_afterpulse:
        local_rates = compute_local_rates(unpacked, spatial_radius)
        n_ap_runs, n_ap_removed = correct_afterpulsing(
            unpacked, local_rates, ap_threshold, rng_seed=base_seed + 9999
        )

    # Stage 2: Hot pixel correction (spatial, per temporal bin)
    if do_hotpixel:
        for bin_idx, t_start in enumerate(range(0, n_frames, bin_size)):
            t_end = min(t_start + bin_size, n_frames)
            unpacked[t_start:t_end], n = correct_bin(
                unpacked[t_start:t_end],
                spatial_radius,
                sigma_threshold,
                rng_seed=base_seed + bin_idx,
            )
            total_hp += n

    return np.packbits(unpacked, axis=2), total_hp, n_ap_runs, n_ap_removed


def main():
    """Fix hot pixels and afterpulsing in the SPAD data named on the command line."""
    parser = argparse.ArgumentParser(description="Fix hot pixels and afterpulsing in SPAD data")
    parser.add_argument("scene_dir", type=str, help="Path to scene directory")
    parser.add_argument(
        "--bin-size",
        type=int,
        default=32,
        help="Temporal bin size for hot pixel detection (default: 32)",
    )
    parser.add_argument(
        "--spatial-radius", type=int, default=7, help="Spatial neighborhood radius (default: 7)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="Hot pixel detection threshold in sigma (default: 5.0)",
    )
    parser.add_argument(
        "--ap-threshold",
        type=float,
        default=1e-6,
        help="Afterpulsing run probability threshold (default: 1e-6)",
    )
    parser.add_argument("--no-hotpixel", action="store_true", help="Skip hot pixel correction")
    parser.add_argument("--no-afterpulse", action="store_true", help="Skip afterpulsing correction")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only count corrections, don't write output"
    )
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    scene_name = scene_dir.name
    npy_files = sorted(scene_dir.glob("*.npy"))
    print(f"Scene: {scene_name} ({len(npy_files)} chunks)")
    print(
        f"Params: bin_size={args.bin_size}, spatial_radius={args.spatial_radius}, "
        f"sigma={args.sigma}, ap_threshold={args.ap_threshold}"
    )
    print(
        f"Stages: afterpulse={'ON' if not args.no_afterpulse else 'OFF'}, "
        f"hotpixel={'ON' if not args.no_hotpixel else 'OFF'}"
    )

    out_dir = scene_dir.parent / f"{scene_name}_corrected"
    if not args.dry_run:
        out_dir.mkdir(exist_ok=True)
        for f in scene_dir.iterdir():
            if f.is_file() and f.suffix != ".npy":
                shutil.copy2(f, out_dir / f.name)
        print(f"Output: {out_dir}")

    grand_hp = 0
    grand_ap_runs = 0
    grand_ap_removed = 0

    for i, npy_path in enumerate(npy_files):
        packed = np.load(str(npy_path))
        fixed, n_hp, n_ap_runs, n_ap_removed = fix_chunk(
            packed,
            args.bin_size,
            args.spatial_radius,
            args.sigma,
            ap_threshold=args.ap_threshold,
            base_seed=(i * 1000),
            do_hotpixel=not args.no_hotpixel,
            do_afterpulse=not args.no_afterpulse,
        )
        grand_hp += n_hp
        grand_ap_runs += n_ap_runs
        grand_ap_removed += n_ap_removed

        label = f"{n_ap_runs} ap_runs, {n_ap_removed} ap_removed, {n_hp} hp"
        if args.dry_run:
            print(f"  {npy_path.name}: {label}")
        else:
            np.save(str(out_dir / npy_path.name), fixed)
            print(f"  {npy_path.name}: {label} ({i + 1}/{len(npy_files)})")

    print(f"\nTotal across {len(npy_files)} chunks:")
    print(f"  Afterpulse: {grand_ap_runs:,} runs corrected, {grand_ap_removed:,} firings removed")
    print(f"  Hot pixel:  {grand_hp:,} pixel-bins corrected")
    if not args.dry_run:
        print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
