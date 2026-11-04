"""Prepare uniform-sampled training data for period-4 corrupted SPAD scenes.

For each scene, prepends an empty frame to each chunk (making 1025 frames),
then samples at step 25 with 25 different offsets (0..24). Each offset
produces a separate .npy file with 41 frames per chunk × 37 chunks = 1517 frames.

Primary offset = 24: indices 24,49,...,999,1024 = original frames 23,48,...,998,1023
  - Skips the empty frame
  - Hits the last original frame (1023)
  - Target for inference = frame[40] (last in each 41-frame group)

Usage:
    python prepare_uniform_data.py
"""

import numpy as np
from pathlib import Path

STEP = 25
PADDED_SIZE = 1025  # 1024 + 1 empty frame at start
SCENES = [
    "D:/Yehe/PC/reconstruction/test/blue-bathroom",
    "D:/Yehe/PC/reconstruction/test/train-passby",
]


def prepare_scene(scene_dir: Path) -> None:
    scene_name = scene_dir.name
    out_dir = scene_dir.parent / f"{scene_name}_step25"
    out_dir.mkdir(exist_ok=True)

    npy_files = sorted(scene_dir.glob("*.npy"))
    n_chunks = len(npy_files)
    print(f"\n=== {scene_name}: {n_chunks} chunks ===")

    # Load all chunks
    chunks = []
    for i, npy_path in enumerate(npy_files):
        data = np.load(str(npy_path))  # (1024, H, W_packed, C)
        chunks.append(data)
        if i == 0:
            print(f"  Chunk shape: {data.shape}, dtype: {data.dtype}")

    _, H, W_packed, C = chunks[0].shape
    empty_frame = np.zeros((1, H, W_packed, C), dtype=chunks[0].dtype)

    # For each offset, sample and save
    for offset in range(STEP):
        # Sample indices in padded (1025-frame) chunk
        padded_indices = np.arange(offset, PADDED_SIZE, STEP)
        n_per_chunk = len(padded_indices)

        # Collect sampled frames from all chunks
        all_frames = []
        for chunk_data in chunks:
            # Conceptually: padded = [empty, chunk_data]
            # padded[0] = zeros, padded[1..1024] = chunk_data[0..1023]
            # So real_idx = padded_idx - 1, and padded_idx=0 -> empty
            for pi in padded_indices:
                if pi == 0:
                    all_frames.append(empty_frame)
                else:
                    real_idx = pi - 1  # map padded -> original
                    all_frames.append(chunk_data[real_idx:real_idx + 1])

        result = np.concatenate(all_frames, axis=0)  # (n_chunks * n_per_chunk, H, W_packed, C)

        out_path = out_dir / f"off{offset:02d}.npy"
        np.save(str(out_path), result)

        tag = " (PRIMARY)" if offset == STEP - 1 else ""
        has_empty = "yes" if offset < 1 else "no"  # offset 0 hits padded index 0
        # Check if last original frame (1023) is included
        hits_last = (PADDED_SIZE - 1) in padded_indices  # padded 1024 = orig 1023
        print(
            f"  off{offset:02d}: {result.shape}, "
            f"padded[{padded_indices[0]}..{padded_indices[-1]}], "
            f"empty={has_empty}, hits_last={hits_last}{tag}"
        )

    print(f"  Saved {STEP} files to {out_dir}")


def main():
    for scene_path in SCENES:
        scene_dir = Path(scene_path)
        if not scene_dir.exists():
            print(f"Skipping {scene_dir} (not found)")
            continue
        prepare_scene(scene_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
