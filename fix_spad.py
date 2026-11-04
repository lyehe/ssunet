"""Fix period-4 temporal repetition in SPAD simulation data.

Root cause: visionsim's old emulate.spad() passed a numpy RNG into a
multi-worker DataLoader. Each worker got a pickled copy of the same RNG
state, so every batch of 4 frames shared identical random thresholds.

Fix strategy — spatial rate estimation:
  1. Group frames by residue class (t % 4). Within each class, frames
     share the same RNG threshold, so only cross-class frames are independent.
  2. Compute per-class mean images (4 independent rate observations per pixel).
  3. Apply spatial Gaussian smoothing to break the 5-level quantization,
     recovering continuous rate estimates while preserving edges.
  4. For animated scenes, estimate per-frame rates by temporal interpolation
     across sliding windows of cross-class observations.
  5. Resample all frames as independent Bernoulli(rate_t) trials.

Usage:
    python fix_spad.py D:/Yehe/PC/reconstruction/test/blue-bathroom
    python fix_spad.py D:/Yehe/PC/reconstruction/test/train-passby
    python fix_spad.py D:/Yehe/PC/reconstruction/test/blue-bathroom --dry-run
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def detect_period4(packed: np.ndarray, n_test_frames: int = 20) -> tuple[float, float]:
    """Check if packed frames repeat with period 4.

    Returns (ratio4, ratio8) — ratio of gap-4/gap-8 diffs to gap-1 diffs.
    The gap-8 check catches cases where gap-4 is masked by scene motion
    but the same-worker correlation persists at gap 8.
    """
    diffs1 = []
    diffs4 = []
    diffs8 = []
    for t in range(min(n_test_frames, packed.shape[0] - 8)):
        diffs1.append(int(np.sum(packed[t] != packed[t + 1])))
        diffs4.append(int(np.sum(packed[t] != packed[t + 4])))
        diffs8.append(int(np.sum(packed[t] != packed[t + 8])))
    mean1 = np.mean(diffs1)
    if mean1 == 0:
        return 1.0, 1.0
    return np.mean(diffs4) / mean1, np.mean(diffs8) / mean1


def compute_anchor_rates(
    unpacked: np.ndarray,
    sigma: float = 2.0,
    n_temporal_windows: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spatially-smoothed rate estimates at temporal anchor points.

    For strongly corrupted data, the naive rate (chunk mean) is quantized
    to {0, 0.25, 0.50, 0.75, 1.00}. Spatial smoothing recovers continuous
    rates by exploiting the fact that neighboring pixels have similar true
    rates but different quantization outcomes.

    Temporal windowing preserves scene animation by estimating rates in
    overlapping windows, rather than collapsing to a single static rate.

    Args:
        unpacked: Binary frames, shape (T, H, W, C).
        sigma: Gaussian smoothing sigma (pixels). Controls spatial pooling.
            Larger = smoother rates, less quantization but more spatial blur.
        n_temporal_windows: Number of temporal windows for rate estimation.
            More windows = better temporal resolution but noisier estimates.

    Returns:
        (anchor_rates, window_centers):
            anchor_rates: shape (n_windows, H, W, C), smoothed rate at each anchor.
            window_centers: shape (n_windows,), frame index of each anchor.
    """
    T, H, W, C = unpacked.shape

    if n_temporal_windows >= T // 4:
        n_temporal_windows = max(1, T // 8)

    window_centers = np.linspace(0, T - 1, n_temporal_windows + 2)[1:-1]
    window_half = T / (2 * n_temporal_windows)

    anchor_rates = np.zeros((n_temporal_windows, H, W, C), dtype=np.float32)

    for i, center in enumerate(window_centers):
        t_start = max(0, int(center - window_half))
        t_end = min(T, int(center + window_half))
        window_mean = unpacked[t_start:t_end].mean(axis=0).astype(np.float32)

        for c in range(C):
            anchor_rates[i, :, :, c] = gaussian_filter(window_mean[:, :, c], sigma=sigma)

    np.clip(anchor_rates, 0.0, 1.0, out=anchor_rates)
    return anchor_rates, window_centers


def interpolate_rate(
    t: int, T: int, anchor_rates: np.ndarray, n_windows: int,
) -> np.ndarray:
    """Linearly interpolate rate for frame t from anchor points."""
    frac = t / max(T - 1, 1) * (n_windows - 1)
    idx_lo = int(np.floor(frac))
    idx_hi = min(idx_lo + 1, n_windows - 1)
    alpha = frac - idx_lo
    if idx_lo == idx_hi:
        return anchor_rates[idx_lo]
    return (1 - alpha) * anchor_rates[idx_lo] + alpha * anchor_rates[idx_hi]


def fix_chunk(
    packed: np.ndarray,
    threshold: float = 0.9,
    seed: int = 0,
    sigma: float = 2.0,
    n_temporal_windows: int = 8,
) -> tuple[np.ndarray, bool, float, float]:
    """Fix period-4 repetition in a single chunk.

    Uses spatial rate estimation to recover continuous per-pixel rates
    from the quantized observations, then resamples all frames as
    independent Bernoulli(rate_t) trials.

    Returns (fixed_packed, was_fixed, ratio4, ratio8).
    """
    ratio4, ratio8 = detect_period4(packed)

    if ratio4 >= threshold and ratio8 >= threshold:
        return packed, False, ratio4, ratio8

    # Unpack to binary
    unpacked = np.unpackbits(packed, axis=2)[:, :, :800, :]
    T, H, W, C = unpacked.shape

    # Estimate spatially-smoothed, temporally-varying rates at anchor points
    anchor_rates, window_centers = compute_anchor_rates(
        unpacked, sigma=sigma, n_temporal_windows=n_temporal_windows,
    )
    n_windows = len(window_centers)

    # Resample all frames as independent Bernoulli(rate_t)
    rng = np.random.RandomState(seed)
    for t in range(T):
        rate_t = interpolate_rate(t, T, anchor_rates, n_windows)
        unpacked[t] = (rng.random((H, W, C)).astype(np.float32) < rate_t).astype(np.uint8)

    return np.packbits(unpacked, axis=2), True, ratio4, ratio8


def main():
    parser = argparse.ArgumentParser(
        description="Fix period-4 temporal repetition in SPAD data"
    )
    parser.add_argument("scene_dir", type=str, help="Path to scene directory")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Period-4 detection threshold (ratio, default: 0.9)")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Spatial Gaussian smoothing sigma (default: 2.0)")
    parser.add_argument("--n-temporal-windows", type=int, default=8,
                        help="Number of temporal windows for rate estimation (default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only detect, don't write output")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    scene_name = scene_dir.name
    npy_files = sorted(scene_dir.glob("*.npy"))
    print(f"Scene: {scene_name} ({len(npy_files)} chunks)")
    print(f"Detection threshold: {args.threshold}")
    print(f"Spatial sigma: {args.sigma}, temporal windows: {args.n_temporal_windows}")

    out_dir = scene_dir.parent / f"{scene_name}_corrected"
    if not args.dry_run:
        out_dir.mkdir(exist_ok=True)
        for f in scene_dir.iterdir():
            if f.is_file() and f.suffix != ".npy":
                shutil.copy2(f, out_dir / f.name)
        print(f"Output: {out_dir}")

    n_fixed = 0
    n_skipped = 0

    for i, npy_path in enumerate(npy_files):
        packed = np.load(str(npy_path))
        fixed, was_fixed, ratio4, ratio8 = fix_chunk(
            packed, threshold=args.threshold, seed=i * 12345,
            sigma=args.sigma, n_temporal_windows=args.n_temporal_windows,
        )

        status = f"FIXED (r4={ratio4:.4f}, r8={ratio8:.4f})" if was_fixed else f"clean (r4={ratio4:.4f}, r8={ratio8:.4f})"
        if was_fixed:
            n_fixed += 1
        else:
            n_skipped += 1

        if args.dry_run:
            print(f"  {npy_path.name}: {status}")
        else:
            np.save(str(out_dir / npy_path.name), fixed)
            print(f"  {npy_path.name}: {status} ({i + 1}/{len(npy_files)})")

    print(f"\nTotal: {n_fixed} chunks fixed, {n_skipped} chunks clean")
    if not args.dry_run:
        print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
