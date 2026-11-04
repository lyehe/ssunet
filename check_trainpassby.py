"""Analyze train-passby scene: compare first 11 frames (affected) vs later frames (good)."""

import numpy as np
from pathlib import Path

# Paths
output_dir = Path("C:/Users/jenkinslab/project/ssunet/output/train-passby")
input_dir = Path("D:/Yehe/PC/reconstruction/test/train-passby")
affected_dir = Path("D:/Yehe/PC/reconstruction/test/train-passby-affected")
corrected_dir = Path("D:/Yehe/PC/reconstruction/test/train-passby_corrected")

print("=" * 80)
print("TRAIN-PASSBY SCENE ANALYSIS")
print("=" * 80)

# 1. Check what models exist
print("\n--- MODEL CHECKPOINTS ---")
models_dir = Path("C:/Users/jenkinslab/project/ssunet/models_spad")
for name in ["train-passby.ckpt", "train-passby-affected.ckpt", "train-passby_corrected.ckpt"]:
    p = models_dir / name
    if p.exists():
        size_mb = p.stat().st_size / 1e6
        print(f"  {name}: {size_mb:.1f} MB")
    else:
        print(f"  {name}: NOT FOUND")

# 2. Check input data structure
print("\n--- INPUT DATA (test chunks) ---")
print(f"  train-passby: {len(list(input_dir.glob('*.npy')))} chunks")
if affected_dir.exists():
    print(f"  train-passby-affected: {len(list(affected_dir.glob('*.npy')))} chunks (first 12, indices 0-11)")
if corrected_dir.exists():
    print(f"  train-passby_corrected: {len(list(corrected_dir.glob('*.npy')))} chunks")

# 3. Analyze input data - compare photon counts in affected vs unaffected chunks
print("\n--- INPUT PHOTON COUNTS (sum per chunk) ---")
chunk_sums = []
for i in range(37):
    fpath = input_dir / f"{i:06d}.npy"
    if fpath.exists():
        data = np.load(str(fpath))
        total = float(data.sum())
        mean = float(data.mean())
        chunk_sums.append((i, total, mean, data.shape, data.dtype))

print(f"  {'Chunk':>6s}  {'Sum':>14s}  {'Mean':>10s}  Shape")
print(f"  {'-----':>6s}  {'---':>14s}  {'----':>10s}  -----")
for i, total, mean, shape, dtype in chunk_sums:
    marker = " <-- AFFECTED" if i < 12 else ""
    print(f"  {i:6d}  {total:14.0f}  {mean:10.6f}  {shape}  {dtype}{marker}")

# 4. Analyze output quality using Laplacian variance and pixel statistics
print("\n--- OUTPUT IMAGE QUALITY (Laplacian variance + stats) ---")

try:
    import cv2
    has_cv2 = True
except ImportError:
    has_cv2 = False
    print("  [cv2 not available, using numpy gradient as proxy for sharpness]")

output_files = sorted(output_dir.glob("*.npy"))
print(f"\n  {'Chunk':>6s}  {'Mean':>10s}  {'Std':>10s}  {'Min':>8s}  {'Max':>10s}  {'Sharpness':>12s}  {'NegFrac':>8s}")
print(f"  {'-----':>6s}  {'----':>10s}  {'---':>10s}  {'---':>8s}  {'---':>10s}  {'---------':>12s}  {'-------':>8s}")

affected_sharpness = []
good_sharpness = []
affected_means = []
good_means = []

for fpath in output_files:
    idx = int(fpath.stem)
    data = np.load(str(fpath))

    mean_val = float(data.mean())
    std_val = float(data.std())
    min_val = float(data.min())
    max_val = float(data.max())
    neg_frac = float((data < 0).mean()) if data.dtype in [np.float32, np.float64] else 0.0

    # Compute sharpness: Laplacian variance on middle slice (or gradient magnitude)
    if len(data.shape) == 3:  # (H, W, C) or (Z, H, W)
        mid = data[data.shape[0] // 2] if data.shape[0] < data.shape[1] else data[:, :, 0]
    elif len(data.shape) == 2:
        mid = data
    else:
        mid = data[0, :, :, 0] if len(data.shape) == 4 else data.reshape(-1)[:256*256].reshape(256,256)

    mid_f32 = mid.astype(np.float32) if mid.dtype != np.float32 else mid

    if has_cv2 and len(mid_f32.shape) == 2:
        lap = cv2.Laplacian(mid_f32, cv2.CV_32F)
        sharpness = float(lap.var())
    else:
        # Gradient-based sharpness
        if len(mid_f32.shape) >= 2:
            gy, gx = np.gradient(mid_f32)
            sharpness = float(np.mean(gx**2 + gy**2))
        else:
            sharpness = 0.0

    marker = " <-- AFFECTED" if idx < 12 else ""
    print(f"  {idx:6d}  {mean_val:10.4f}  {std_val:10.4f}  {min_val:8.4f}  {max_val:10.4f}  {sharpness:12.4f}  {neg_frac:8.4f}{marker}")

    if idx < 12:
        affected_sharpness.append(sharpness)
        affected_means.append(mean_val)
    else:
        good_sharpness.append(sharpness)
        good_means.append(mean_val)

print("\n--- SUMMARY ---")
if affected_sharpness and good_sharpness:
    print(f"  Affected chunks (0-11):")
    print(f"    Avg sharpness: {np.mean(affected_sharpness):.4f}")
    print(f"    Avg mean pixel: {np.mean(affected_means):.4f}")
    print(f"  Good chunks (12-36):")
    print(f"    Avg sharpness: {np.mean(good_sharpness):.4f}")
    print(f"    Avg mean pixel: {np.mean(good_means):.4f}")
    print(f"  Sharpness ratio (affected/good): {np.mean(affected_sharpness)/np.mean(good_sharpness):.4f}")
    print(f"  Mean ratio (affected/good): {np.mean(affected_means)/np.mean(good_means):.4f}")

# 5. Compare affected input chunks vs corrected input chunks
print("\n--- CORRECTION ANALYSIS (original vs corrected input) ---")
if corrected_dir.exists():
    diffs = []
    for i in range(min(12, len(list(corrected_dir.glob("*.npy"))))):
        orig = np.load(str(input_dir / f"{i:06d}.npy"))
        corr_path = corrected_dir / f"{i:06d}.npy"
        if corr_path.exists():
            corr = np.load(str(corr_path))
            diff_count = int(np.sum(orig != corr))
            orig_sum = int(orig.sum())
            corr_sum = int(corr.sum())
            diffs.append((i, diff_count, orig_sum, corr_sum))
            print(f"  Chunk {i:3d}: {diff_count:>10d} pixels differ, orig_sum={orig_sum:>12d}, corr_sum={corr_sum:>12d}, delta={corr_sum-orig_sum:>10d}")

    # Also check if chunks 12+ have corrections
    for i in range(12, 37):
        corr_path = corrected_dir / f"{i:06d}.npy"
        if corr_path.exists():
            orig = np.load(str(input_dir / f"{i:06d}.npy"))
            corr = np.load(str(corr_path))
            diff_count = int(np.sum(orig != corr))
            if diff_count > 0:
                print(f"  Chunk {i:3d}: {diff_count:>10d} pixels differ (UNEXPECTED)")
            else:
                pass  # no diff, skip printing for clean output
    print(f"  Chunks 12-36: checking for corrections...")
    any_diff = False
    for i in range(12, 37):
        corr_path = corrected_dir / f"{i:06d}.npy"
        if corr_path.exists():
            orig = np.load(str(input_dir / f"{i:06d}.npy"))
            corr = np.load(str(corr_path))
            diff_count = int(np.sum(orig != corr))
            if diff_count > 0:
                any_diff = True
    if not any_diff:
        print(f"  Chunks 12-36: all have corrections too (from fix log)")

# 6. Side-by-side frame 5 vs frame 15 from output
print("\n--- FRAME 5 vs FRAME 15 COMPARISON ---")
for idx in [5, 15]:
    npy_path = output_dir / f"{idx:06d}.npy"
    if npy_path.exists():
        data = np.load(str(npy_path))
        print(f"  Frame {idx}: shape={data.shape}, dtype={data.dtype}")
        print(f"    mean={data.mean():.4f}, std={data.std():.4f}, min={data.min():.4f}, max={data.max():.4f}")
        # Per-channel stats if RGB
        if len(data.shape) >= 3:
            c_axis = -1 if data.shape[-1] in [1,3] else 0
            if data.shape[-1] == 3:
                for c, cname in enumerate(["R", "G", "B"]):
                    ch = data[..., c]
                    print(f"    {cname}: mean={ch.mean():.4f}, std={ch.std():.4f}")

print("\n" + "=" * 80)
print("DONE")
