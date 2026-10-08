"""Dataset and loading for preprocessed magnitude slices.

GIVEN. Reads the ``.npz`` shards written by ``data/preprocess.py`` and yields
batches of 128x128 magnitude slices, normalised **per volume** to [0, 1].

**Normalisation and noise.** Each volume is divided by its 99.5th-percentile
intensity and clipped to [0, 1] (:func:`normalise_volume`). Per volume, not per
slice: the noise level is nearly constant within a volume, so every slice of it
keeps the same noise std, and mostly-air edge slices stay dark instead of having
their noise stretched to full scale. The measured noise std of each slice, in
these normalised units, is ``dataset.sigmas`` -- the VAE likelihood uses it. No
patient data ships with the repo; this only touches files the student created
locally from their own fastMRI download (see data/REGISTER_FIRST.md).

**Held-out split.** A prior must be evaluated on slices it never saw. Volumes
named in :data:`HELDOUT_VOLUMES` are the *test* split; everything else is
*train*. (Deliberately no third validation split for a one-week school; if you
tune hyperparameters hard, know you are tuning on the training volumes.) The
pre-trained checkpoint was trained with ``split="train"`` (see
CHECKPOINTS.md), so ``FastMRISlices(root, split="test")`` gives you slices that
are held out from the checkpoint too. Train on "train", tune on "train",
report numbers on "test" -- and say so in your table.

**On the school GPU server** the data is prepared once into split
sub-directories (``scripts/server_prepare_data.py``)::

    processed/train/*.npz   # most of fastMRI singlecoil_train
    processed/val/*.npz     # the rest of singlecoil_train, split by patient
    processed/test/*.npz    # fastMRI singlecoil_val -- the final test set

When ``root`` has these sub-directories, ``split`` simply picks one of them and
``heldout`` is ignored. There *is* a validation split here: tune on "val",
report on "test" once.
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


#: Intensity percentile a volume is scaled by (robust to a few hot pixels).
VOLUME_PERCENTILE = 99.5


def normalise_volume(
    vol: np.ndarray, noise_sigma: float
) -> tuple[np.ndarray, float, float]:
    """Scale a (n, H, W) volume to [0, 1]; return ``(vol_norm, scale, sigma_norm)``.

    GIVEN. ``scale`` is the volume's 99.5th-percentile intensity; the brightest
    0.5% of pixels are clipped to 1. ``sigma_norm = noise_sigma / scale`` is the
    noise std in the normalised units -- the same for every slice of the volume.
    As with :func:`normalise`, keep ``scale`` to map results back
    (CLAUDE.md Contract 2).
    """
    vol = np.asarray(vol, dtype=np.float32)
    scale = float(np.percentile(vol, VOLUME_PERCENTILE))
    if scale <= 0.0:
        scale = 1.0
    return np.clip(vol / np.float32(scale), 0.0, 1.0), scale, float(noise_sigma) / scale


def denormalise(x_norm: np.ndarray, scale: float) -> np.ndarray:
    """Invert :func:`normalise`: map a [0, 1] image back by ``scale``. GIVEN."""
    return np.asarray(x_norm, dtype=np.float32) * np.float32(scale)


#: Volumes reserved for evaluation. The first volumes in the fastMRI archive, so
#: every download (``pixi run download --n 2`` or more) contains them. The mentor's
#: checkpoint excludes them; never train on them.
HELDOUT_VOLUMES: tuple[str, ...] = ("file1000593", "file1002067")

#: Sub-directory names of the server layout (see the module docstring).
SPLIT_DIRS: tuple[str, ...] = ("train", "val", "test")


class FastMRISlices:
    """In-memory dataset of preprocessed magnitude slices.

    Args:
        root: directory containing ``*.npz`` shards, each with key ``slices``
            of shape (n, H, W).
        normalize: if True, scale each volume to [0, 1] (:func:`normalise_volume`).
        split: ``None`` (every shard), ``"train"`` (shards not in ``heldout``)
            or ``"test"`` (shards in ``heldout``). ``"val"`` is accepted as the
            same held-out shards: a laptop download is too small for three
            splits, so notebooks can ask for "val" and run anywhere. With the server layout
            (``root/train``, ``root/val``, ``root/test``) it is one of those
            three, and ``None`` is not allowed -- it would mix in the test set.
        heldout: volume names (shard stems) that form the test split. Ignored
            with the server layout.
    """

    def __init__(
        self,
        root: str | Path,
        normalize: bool = True,
        split: str | None = None,
        heldout: tuple[str, ...] = HELDOUT_VOLUMES,
    ):
        root = Path(root)
        if any((root / d).is_dir() for d in SPLIT_DIRS):
            shards = self._split_dir_shards(root, split)
        else:
            shards = self._heldout_shards(root, split, heldout)
        self.split = split
        self.volumes = [s.stem for s in shards]
        arrays, noise = [], []
        for shard in shards:
            with np.load(shard) as z:
                if "noise_sigma" not in z:
                    raise KeyError(
                        f"{shard} has no 'noise_sigma': it was preprocessed by an older "
                        f"version. Re-run `pixi run preprocess` (laptop) or "
                        f"`pixi run server-prepare --force` (server)."
                    )
                arrays.append(z["slices"])
                noise.append(float(z["noise_sigma"]))
        # which volume each slice came from (index into self.volumes), for reporting
        self.volume_index = np.concatenate(
            [np.full(len(a), i, dtype=np.int32) for i, a in enumerate(arrays)]
        )
        # Per-slice scale (shared within a volume) maps a reconstruction back to
        # the original intensity range (CLAUDE.md Contract 2); 1.0 when not
        # normalising, so denormalise is always a valid inverse. ``sigmas`` is
        # each slice's noise std in the same units as ``slices``.
        scales, sigmas = [], []
        for i, (a, nz) in enumerate(zip(arrays, noise)):
            scale = 1.0
            if normalize:
                arrays[i], scale, nz = normalise_volume(a, nz)
            scales.append(np.full(len(a), scale, dtype=np.float32))
            sigmas.append(np.full(len(a), nz, dtype=np.float32))
        self.slices = np.concatenate(arrays, axis=0).astype(np.float32)
        self.scales = np.concatenate(scales)
        self.sigmas = np.concatenate(sigmas)

    @staticmethod
    def _split_dir_shards(root: Path, split: str | None) -> list[Path]:
        """Server layout: one sub-directory per split."""
        if split not in SPLIT_DIRS:
            raise ValueError(
                f"{root} has split sub-directories, so split must be one of "
                f"{list(SPLIT_DIRS)}, got {split!r}."
            )
        shards = sorted((root / split).glob("*.npz"))
        if not shards:
            raise FileNotFoundError(
                f"No .npz shards in {root / split}. On the server, run "
                f"`pixi run server-setup` (see SERVER.md)."
            )
        return shards

    @staticmethod
    def _heldout_shards(root: Path, split: str | None, heldout: tuple[str, ...]) -> list[Path]:
        """Laptop layout: one flat directory; the test split is named volumes."""
        shards = sorted(root.glob("*.npz"))
        if not shards:
            raise FileNotFoundError(
                f"No .npz shards in {root}. Run `pixi run download` then "
                f"`pixi run preprocess` first (see data/REGISTER_FIRST.md)."
            )
        available = [s.stem for s in shards]
        if split == "train":
            shards = [s for s in shards if s.stem not in heldout]
        elif split in ("val", "test"):  # too little laptop data for both
            shards = [s for s in shards if s.stem in heldout]
        elif split is not None:
            raise ValueError(f"split must be None, 'train', 'val' or 'test', got {split!r}")
        if not shards:
            raise FileNotFoundError(
                f"No volumes for split={split!r}. Held-out volumes are {list(heldout)}; "
                f"you have {available}. Download more volumes (`pixi run download --n 4`) "
                f"or pass `heldout=` explicitly."
            )
        return shards

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.slices[i]


def data_loader(
    dataset, batch_size: int, *, shuffle: bool = True, seed: int = 0, with_sigma: bool = False
):
    """Yield batches of shape (batch_size, H, W) for one epoch.

    With ``with_sigma=True`` yield ``(batch, sigma)`` instead, ``sigma`` being the
    (batch_size,) noise std of each slice (``dataset.sigmas``). Drops the last
    partial batch so shapes stay static for JIT.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n - batch_size + 1, batch_size):
        batch = idx[start : start + batch_size]
        images = np.stack([dataset[i] for i in batch], axis=0)
        yield (images, dataset.sigmas[batch]) if with_sigma else images
