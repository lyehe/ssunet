"""Test RNG behavior across multiple batches in DataLoader workers."""
import functools
import os

import numpy as np
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __len__(self):
        return 40  # 10 batches of 4

    def __getitem__(self, idx):
        return idx, np.ones((2, 2))


def collate(batch, *, rng):
    idxs, imgs = zip(*batch)
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
        print(f"  frames {frames[0]:2d}-{frames[-1]:2d}: {randoms[0]:.6f} {randoms[1]:.6f} {randoms[2]:.6f} {randoms[3]:.6f}  pid={pid}")
