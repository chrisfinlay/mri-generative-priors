"""Preprocess fastMRI single-coil .h5 volumes into 128x128 magnitude .npz shards.

Reads the ``.h5`` files in ``data/raw/`` and, for each, takes the provided
``reconstruction_esc`` dataset — the *emulated single-coil* reconstruction,
which is already the real-valued magnitude image (320x320 float32, the standard
fastMRI target) — centre-crops it to a square, resizes to 128x128, and writes
one ``.npz`` per volume with key ``slices`` of shape (n_slices, 128, 128).

We use ``reconstruction_esc`` (not the raw ``kspace``) for the magnitude path,
per CLAUDE.md Contract 3 / the data rules: ``kspace`` is the *native* size
(e.g. 640x368) with readout oversampling, so ``ifft2c(kspace)`` does not equal
``reconstruction_esc`` — the standard target is the cropped esc image. The raw
``kspace`` is reserved for the complex (2-channel) stretch path.

    python data/preprocess.py --size 128

No patient data is committed; this only reads files you downloaded yourself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from skimage.transform import resize

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
OUT_DIR = HERE / "processed"

SOURCE_KEY = "reconstruction_esc"


def _centre_crop_square(img: np.ndarray) -> np.ndarray:
    """Centre-crop a 2D image to its largest centred square (no-op if square)."""
    h, w = img.shape
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return img[top : top + s, left : left + s]


def _to_slices(esc: np.ndarray, size: int) -> np.ndarray:
    """(n, H, W) real magnitude volume -> (n, size, size) float32, square-cropped."""
    esc = np.abs(np.asarray(esc, dtype=np.float32))  # |.|: no-op for magnitude, safe
    out = np.empty((esc.shape[0], size, size), dtype=np.float32)
    for i, sl in enumerate(esc):
        out[i] = resize(
            _centre_crop_square(sl), (size, size), anti_aliasing=True, preserve_range=True
        )
    return out


def preprocess(size: int = 128) -> None:
    vols = sorted(RAW_DIR.glob("*.h5"))
    if not vols:
        raise SystemExit(
            f"No .h5 files in {RAW_DIR}. Run `python data/download_subset.py` first."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for vol in vols:
        with h5py.File(vol, "r") as f:
            if SOURCE_KEY not in f:
                print(f"skip {vol.name}: no '{SOURCE_KEY}' dataset")
                continue
            esc = f[SOURCE_KEY][()]  # (n_slices, 320, 320) float32 magnitude
        slices = _to_slices(esc, size)
        out = OUT_DIR / f"{vol.stem}.npz"
        np.savez_compressed(out, slices=slices)
        print(f"{vol.name}: {slices.shape[0]} slices -> {out}")

    total = sum(np.load(p)["slices"].shape[0] for p in OUT_DIR.glob("*.npz"))
    print(f"done: {total} slices in {OUT_DIR}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=128)
    preprocess(p.parse_args().size)


if __name__ == "__main__":
    main()
