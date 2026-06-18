"""Preprocess fastMRI .h5 volumes into 128x128 magnitude-slice .npz shards.

Reads the single-coil .h5 files in data/raw/, reconstructs each slice from its
fully-sampled k-space with the centred inverse FFT, takes the magnitude,
centre-crops/resizes to 128x128, and writes one .npz shard per volume with key
``slices`` of shape (n_slices, 128, 128).

    python data/preprocess.py --size 128

No patient data is committed; this only reads files you downloaded yourself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from skimage.transform import resize

from mrigen.fourier import ifft2c

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
OUT_DIR = HERE / "processed"


def _to_magnitude(kspace: np.ndarray) -> np.ndarray:
    """(n, H, W) complex single-coil k-space -> (n, H, W) magnitude images."""
    img = np.asarray(ifft2c(kspace))
    return np.abs(img)


def _resize_stack(mag: np.ndarray, size: int) -> np.ndarray:
    out = np.empty((mag.shape[0], size, size), dtype=np.float32)
    for i, sl in enumerate(mag):
        out[i] = resize(sl, (size, size), anti_aliasing=True, preserve_range=True)
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
            if "kspace" not in f:
                print(f"skip {vol.name}: no 'kspace' dataset")
                continue
            kspace = f["kspace"][()]  # (n_slices, H, W) complex
        mag = _to_magnitude(kspace)
        slices = _resize_stack(mag, size)
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
