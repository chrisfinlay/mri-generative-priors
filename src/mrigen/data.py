"""Dataset and loading for preprocessed magnitude slices.

GIVEN. Reads the ``.npz`` shards written by ``data/preprocess.py`` and yields
batches of 128x128 magnitude slices, per-slice normalised to [0, 1]. No
patient data ships with the repo; this only touches files the student created
locally from their own fastMRI download (see data/REGISTER_FIRST.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FastMRISlices:
    """In-memory dataset of preprocessed magnitude slices.

    Args:
        root: directory containing ``*.npz`` shards, each with key ``slices``
            of shape (n, H, W).
        normalize: if True, scale each slice to [0, 1] by its own max.
    """

    def __init__(self, root: str | Path, normalize: bool = True):
        root = Path(root)
        shards = sorted(root.glob("*.npz"))
        if not shards:
            raise FileNotFoundError(
                f"No .npz shards in {root}. Run `pixi run download` then "
                f"`pixi run preprocess` first (see data/REGISTER_FIRST.md)."
            )
        arrays = [np.load(s)["slices"] for s in shards]
        self.slices = np.concatenate(arrays, axis=0).astype(np.float32)
        if normalize:
            maxes = self.slices.reshape(len(self.slices), -1).max(axis=1)
            maxes = np.where(maxes > 0, maxes, 1.0)
            self.slices = self.slices / maxes[:, None, None]

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.slices[i]


def data_loader(dataset, batch_size: int, *, shuffle: bool = True, seed: int = 0):
    """Yield batches of shape (batch_size, H, W) for one epoch.

    Drops the last partial batch so shapes stay static for JIT.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n - batch_size + 1, batch_size):
        batch = idx[start : start + batch_size]
        yield np.stack([dataset[i] for i in batch], axis=0)
