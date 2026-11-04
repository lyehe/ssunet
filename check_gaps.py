"""Check frame correlation at various gaps for all chunks in a scene."""

import sys
from pathlib import Path

import numpy as np

scene_dir = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/Yehe/PC/reconstruction/test/train-passby")
)
npy_files = sorted(scene_dir.glob("*.npy"))

print(f"Scene: {scene_dir.name} ({len(npy_files)} chunks)")
print(
    f"{'Chunk':>6} | {'gap-1':>8} {'gap-4':>8} {'gap-8':>8} {'gap-16':>8} {'gap-32':>8} | "
    f"{'r4':>6} {'r8':>6} {'r16':>6} {'r32':>6}"
)
print("-" * 95)

for i, npy_path in enumerate(npy_files):
    packed = np.load(str(npy_path))
    unpacked = np.unpackbits(packed[:64], axis=2)[:, :, :800, :]

    gaps = {}
    for gap in [1, 4, 8, 16, 32]:
        diffs = []
        for t in range(min(20, 64 - gap)):
            diff_val = float(np.mean(unpacked[t] != unpacked[t + gap]))
            diffs.append(diff_val)
        gaps[gap] = np.mean(diffs)

    g1 = gaps[1]
    if g1 == 0:
        ratios = "   n/a    n/a    n/a    n/a"
    else:
        ratios = (
            f"{gaps[4] / g1:6.3f} {gaps[8] / g1:6.3f} {gaps[16] / g1:6.3f} {gaps[32] / g1:6.3f}"
        )

    print(
        f"{i:6d} | {gaps[1]:8.4f} {gaps[4]:8.4f} {gaps[8]:8.4f} {gaps[16]:8.4f} "
        f"{gaps[32]:8.4f} | {ratios}"
    )
