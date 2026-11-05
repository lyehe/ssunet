"""Analyze SPAD data for fixed pattern noise and frame correlations."""

from pathlib import Path

import numpy as np

DATA_DIR = Path("D:/Yehe/PC/reconstruction/test/blue-bathroom")
NUM_FRAMES = 32

# ── 1. Load first file, print shape and dtype ──────────────────────────────
print("=" * 70)
print("1. LOAD FIRST .npy FILE")
print("=" * 70)
first_file = DATA_DIR / "000000.npy"
frame0 = np.load(first_file)
print(f"File: {first_file}")
print(f"Shape: {frame0.shape}")
print(f"Dtype: {frame0.dtype}")
print(f"Min: {frame0.min()}, Max: {frame0.max()}")
print(f"Unique values: {np.unique(frame0)[:20]}{'...' if len(np.unique(frame0)) > 20 else ''}")
print(f"Number of unique values: {len(np.unique(frame0))}")
print()

# ── 2. Single spatial location across 32 frames ────────────────────────────
print("=" * 70)
print("2. RAW VALUES AT A SINGLE PIXEL ACROSS 32 FRAMES")
print("=" * 70)

# Load all 32 frames
frames = []
for i in range(NUM_FRAMES):
    f = np.load(DATA_DIR / f"{i:06d}.npy")
    frames.append(f)

# Determine indexing based on shape
ndim = frame0.ndim
print(f"Frame ndim: {ndim}, shape: {frame0.shape}")

if ndim == 2:
    h_idx, w_idx = min(400, frame0.shape[0] - 1), min(50, frame0.shape[1] - 1)
    values = [frames[i][h_idx, w_idx] for i in range(NUM_FRAMES)]
    print(f"Pixel location: h={h_idx}, w={w_idx}")
elif ndim == 3:
    h_idx = min(400, frame0.shape[0] - 1)
    w_idx = min(50, frame0.shape[1] - 1)
    c_idx = 0
    # Try channels-last (HWC), then channels-first (CHW) layout
    if frame0.shape[-1] <= 4:  # likely (H, W, C)
        values = [frames[i][h_idx, w_idx, c_idx] for i in range(NUM_FRAMES)]
        print(f"Pixel location: h={h_idx}, w={w_idx}, c={c_idx} (H,W,C layout)")
    else:  # likely (C, H, W)
        h_idx = min(400, frame0.shape[1] - 1)
        w_idx = min(50, frame0.shape[2] - 1)
        values = [frames[i][c_idx, h_idx, w_idx] for i in range(NUM_FRAMES)]
        print(f"Pixel location: c={c_idx}, h={h_idx}, w={w_idx} (C,H,W layout)")
else:
    # 4D or other - just take a reasonable slice
    print(f"Unexpected ndim={ndim}, using first elements")
    values = [frames[i].flat[50 * frame0.shape[-1] + 400] for i in range(NUM_FRAMES)]

print("Raw uint8 values for frames 0-31:")
for i in range(0, NUM_FRAMES, 8):
    chunk = values[i : i + 8]
    print(f"  Frames {i:2d}-{i + len(chunk) - 1:2d}: {chunk}")

all_identical = all(v == values[0] for v in values)
print(f"All identical? {all_identical}")
# Check for repeating patterns
for period in [2, 4, 8, 16]:
    is_periodic = all(values[i] == values[i % period] for i in range(NUM_FRAMES))
    if is_periodic:
        print(f"  Repeating pattern with period {period} detected!")
print()

# ── 3. Unpack bits & frame-to-frame correlation (4x4 patch) ────────────────
print("=" * 70)
print("3. BIT-UNPACKED PATCH CORRELATION MATRIX (frames 0-31)")
print("=" * 70)

# Unpack bits from the frames at a small patch
# Choose patch based on dimensionality
if ndim == 2:
    h_start, w_start = 200, 25
    patch_h, patch_w = 4, 4
    patches_packed = np.array(
        [
            frames[i][h_start : h_start + patch_h, w_start : w_start + patch_w]
            for i in range(NUM_FRAMES)
        ]
    )
elif ndim == 3:
    if frame0.shape[-1] <= 4:
        h_start, w_start = 200, 25
        patch_h, patch_w = 4, 4
        patches_packed = np.array(
            [
                frames[i][h_start : h_start + patch_h, w_start : w_start + patch_w, 0]
                for i in range(NUM_FRAMES)
            ]
        )
    else:
        h_start, w_start = min(200, frame0.shape[1] - 5), min(25, frame0.shape[2] - 5)
        patch_h, patch_w = 4, 4
        patches_packed = np.array(
            [
                frames[i][0, h_start : h_start + patch_h, w_start : w_start + patch_w]
                for i in range(NUM_FRAMES)
            ]
        )
else:
    # fallback
    patches_packed = np.array([frames[i].flat[:16].reshape(4, 4) for i in range(NUM_FRAMES)])

print(f"Packed patch shape per frame: {patches_packed.shape[1:]}")
print(f"Packed patch dtype: {patches_packed.dtype}")

# Unpack bits
patches_unpacked = np.unpackbits(patches_packed, axis=-1)  # unpack along last axis
print(f"Unpacked patch shape per frame: {patches_unpacked.shape[1:]}")

# Flatten each frame's patch to a vector
n_frames = patches_unpacked.shape[0]
vectors = patches_unpacked.reshape(n_frames, -1).astype(np.float64)
print(f"Vector length per frame: {vectors.shape[1]}")

# Correlation matrix
corr_matrix = np.corrcoef(vectors)
print("\nCorrelation matrix (frames 0-31):")
print("     ", "  ".join(f"{i:5d}" for i in range(min(16, NUM_FRAMES))))
for i in range(min(16, NUM_FRAMES)):
    row = corr_matrix[i, : min(16, NUM_FRAMES)]
    print(f"{i:3d}: ", "  ".join(f"{v:5.3f}" for v in row))

# Check for identical or very high correlation
print("\nIdentical frame pairs (corr == 1.0, i != j):")
identical_pairs = []
for i in range(NUM_FRAMES):
    for j in range(i + 1, NUM_FRAMES):
        if np.isclose(corr_matrix[i, j], 1.0, atol=1e-10):
            identical_pairs.append((i, j))
if identical_pairs:
    for p in identical_pairs[:20]:
        print(f"  Frame {p[0]} and Frame {p[1]}")
else:
    print("  None found")

print("\nHighly correlated pairs (corr > 0.9, i != j):")
high_corr = []
for i in range(NUM_FRAMES):
    for j in range(i + 1, NUM_FRAMES):
        if corr_matrix[i, j] > 0.9:
            high_corr.append((i, j, corr_matrix[i, j]))
if high_corr:
    for p in sorted(high_corr, key=lambda x: -x[2])[:20]:
        print(f"  Frame {p[0]} and Frame {p[1]}: corr = {p[2]:.6f}")
else:
    print("  None found")

# Statistics
off_diag = corr_matrix[np.triu_indices(NUM_FRAMES, k=1)]
print("\nCorrelation stats (off-diagonal):")
print(f"  Mean: {off_diag.mean():.6f}")
print(f"  Std:  {off_diag.std():.6f}")
print(f"  Min:  {off_diag.min():.6f}")
print(f"  Max:  {off_diag.max():.6f}")
print()

# ── 4. Per-frame mean across all spatial pixels ────────────────────────────
print("=" * 70)
print("4. PER-FRAME MEAN (average over all spatial pixels)")
print("=" * 70)

per_frame_means = []
for i in range(NUM_FRAMES):
    # Unpack all bits and compute mean
    unpacked = np.unpackbits(frames[i])
    per_frame_means.append(unpacked.mean())

per_frame_means = np.array(per_frame_means)
print("Frame means (fraction of 1-bits):")
for i in range(0, NUM_FRAMES, 8):
    chunk = per_frame_means[i : i + 8]
    labels = [f"F{j:02d}={v:.6f}" for j, v in zip(range(i, i + len(chunk)), chunk, strict=True)]
    print(f"  {', '.join(labels)}")

print(f"\nMean of means: {per_frame_means.mean():.6f}")
print(f"Std of means:  {per_frame_means.std():.6f}")
print(f"Min: {per_frame_means.min():.6f}, Max: {per_frame_means.max():.6f}")
print(f"Range: {per_frame_means.max() - per_frame_means.min():.6f}")
is_constant = per_frame_means.std() < 1e-4
print(f"Approximately constant? {is_constant}")
print()

# ── 5. Autocorrelation of per-frame means ──────────────────────────────────
print("=" * 70)
print("5. AUTOCORRELATION OF PER-FRAME MEANS")
print("=" * 70)

centered = per_frame_means - per_frame_means.mean()
norm = np.sum(centered**2)
if norm > 0:
    autocorr = np.correlate(centered, centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]  # positive lags only
    autocorr = autocorr / norm  # normalize
    print("Autocorrelation at lag k:")
    for k in range(min(16, len(autocorr))):
        bar = "#" * int(abs(autocorr[k]) * 40)
        sign = "+" if autocorr[k] >= 0 else "-"
        print(f"  Lag {k:2d}: {autocorr[k]:+.4f}  {sign}{bar}")

    # Find peaks
    peaks = []
    for k in range(1, len(autocorr) - 1):
        if autocorr[k] > autocorr[k - 1] and autocorr[k] > autocorr[k + 1] and autocorr[k] > 0.1:
            peaks.append((k, autocorr[k]))
    if peaks:
        print("\n  Autocorrelation peaks (lag > 0, value > 0.1):")
        for lag, val in peaks:
            print(f"    Lag {lag}: {val:.4f}")
    else:
        print("\n  No significant periodic peaks detected.")
else:
    print("  All frame means are identical - no variation to analyze.")
print()

# ── 6. Temporal offset similarity ──────────────────────────────────────────
print("=" * 70)
print("6. FRAME SIMILARITY AT DIFFERENT TEMPORAL OFFSETS")
print("=" * 70)


# Use packed bytes for fast comparison
def frame_similarity(a, b):
    """Fraction of matching bits between two frames."""
    xor = np.bitwise_xor(a, b)
    # Count differing bits
    diff_bits = sum(bin(byte).count("1") for byte in xor.flat)
    total_bits = a.size * 8
    return 1.0 - diff_bits / total_bits


# Use numpy popcount approach for speed
def frame_similarity_fast(a, b):
    """Fraction of matching bits using lookup table."""
    xor = np.bitwise_xor(a, b)
    # Popcount via lookup table
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)
    diff_bits = lut[xor.ravel()].sum()
    total_bits = a.size * 8
    return 1.0 - diff_bits / total_bits


print("Similarity between frame[0] and frame[k]:")
ref = frames[0]
for k in range(1, min(17, NUM_FRAMES)):
    sim = frame_similarity_fast(ref, frames[k])
    bar = "#" * int(sim * 50)
    print(f"  frame[0] vs frame[{k:2d}]: similarity = {sim:.6f}  {bar}")

print()
print("Similarity matrix for specific offset comparisons:")
offsets_to_check = [1, 2, 4, 8, 16]
for offset in offsets_to_check:
    if offset >= NUM_FRAMES:
        continue
    sims = []
    for i in range(min(NUM_FRAMES - offset, 16)):
        sims.append(frame_similarity_fast(frames[i], frames[i + offset]))
    mean_sim = np.mean(sims)
    std_sim = np.std(sims)
    print(f"  Offset {offset:2d}: mean similarity = {mean_sim:.6f} +/- {std_sim:.6f}")

print()
print("Direct comparison: frame[0] vs frame[1] and frame[0] vs frame[4]:")
sim_01 = frame_similarity_fast(frames[0], frames[1])
sim_04 = frame_similarity_fast(frames[0], frames[4])
print(f"  frame[0] vs frame[1]: {sim_01:.6f}")
print(f"  frame[0] vs frame[4]: {sim_04:.6f}")
if sim_04 > sim_01:
    print(f"  -> frame[0] and frame[4] are MORE similar (by {sim_04 - sim_01:.6f})")
elif sim_01 > sim_04:
    print(f"  -> frame[0] and frame[1] are MORE similar (by {sim_01 - sim_04:.6f})")
else:
    print("  -> Equal similarity")

# Also check if any frames are byte-identical
print()
print("Checking for byte-identical frame pairs (first 32 frames):")
identical_found = False
for i in range(NUM_FRAMES):
    for j in range(i + 1, NUM_FRAMES):
        if np.array_equal(frames[i], frames[j]):
            print(f"  Frame {i} and Frame {j} are BYTE-IDENTICAL!")
            identical_found = True
if not identical_found:
    print("  No byte-identical frames found.")

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
