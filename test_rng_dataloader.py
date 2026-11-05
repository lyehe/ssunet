"""Test RNG behavior across multiple batches in DataLoader workers."""

import functools
import os

import numpy as np
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    """Forty samples, each an index with a constant 2x2 image."""

    def __len__(self):
        """Ten batches of four samples."""
        return 40  # 10 batches of 4

    def __getitem__(self, idx):
        """The index and a constant 2x2 image."""
        return idx, np.ones((2, 2))


def collate(batch, *, rng):
    """Indices, one draw from rng per sample, and the collating process id."""
    idxs, _imgs = zip(*batch, strict=True)
    randoms = [float(rng.random()) for _ in idxs]
    return list(idxs), randoms, os.getpid()


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    loader = DataLoader(
        DummyDataset(),
        batch_size=4,
        num_workers=4,
        collate_fn=functools.partial(collate, rng=rng),
        persistent_workers=False,
    )

    for idxs, randoms, pid in loader:
        frames = [int(i) for i in idxs]
        print(
            f"  frames {frames[0]:2d}-{frames[-1]:2d}: {randoms[0]:.6f} {randoms[1]:.6f} "
            f"{randoms[2]:.6f} {randoms[3]:.6f}  pid={pid}"
        )
