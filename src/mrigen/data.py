"""Dataset and loading for preprocessed magnitude slices.

GIVEN. Reads the ``.npz`` shards written by ``data/preprocess.py`` and yields
batches of 128x128 magnitude slices, per-slice normalised to [0, 1]. No
patient data ships with the repo; this only touches files the student created
locally from their own fastMRI download (see data/REGISTER_FIRST.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def normalise(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale a magnitude image to [0, 1] by its max; return ``(x_norm, scale)``.

    GIVEN. Per CLAUDE.md Contract 2, normalisation must be applied *identically*
    at training and reconstruction time, so we **return the scale** rather than
    discarding it: keep it next to the measurement and use ``denormalise`` to map
    a reconstruction back to the original intensity range. A non-positive max
    (e.g. an all-zero slice) falls back to scale 1.0 so the round-trip is safe.
    """
    x = np.asarray(x, dtype=np.float32)
    scale = float(x.max())
    if scale <= 0.0:
        scale = 1.0
    return x / np.float32(scale), scale


def denormalise(x_norm: np.ndarray, scale: float) -> np.ndarray:
    """Invert :func:`normalise`: map a [0, 1] image back by ``scale``. GIVEN."""
    return np.asarray(x_norm, dtype=np.float32) * np.float32(scale)


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
        # Per-slice scales kept so a reconstruction can be mapped back to the
        # original intensity range (CLAUDE.md Contract 2). Scale is 1.0 when not
        # normalising, so denormalise is always a valid inverse.
        self.scales = np.ones(len(self.slices), dtype=np.float32)
        if normalize:
            for i in range(len(self.slices)):
                self.slices[i], self.scales[i] = normalise(self.slices[i])

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
