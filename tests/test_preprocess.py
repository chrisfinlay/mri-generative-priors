"""Preprocessing (GIVEN): the per-volume noise estimate behind the VAE likelihood."""

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from preprocess import estimate_noise_sigma, preprocess_volume


def _ifft2c(k):
    # numpy twin of mrigen.fourier.ifft2c (CLAUDE.md Contract 1), to build a fake volume
    ax = (-2, -1)
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=ax), norm="ortho"), axes=ax)


def _fake_volume(sigma, n=2, rows=640, cols=320, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((n, rows, cols))
    img[:, 200:440, 60:260] = 50.0  # bright flat block, smooth -> low-frequency k-space
    ax = (-2, -1)
    k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=ax), norm="ortho"), axes=ax)
    k = k + sigma * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape))
    return k.astype(np.complex64)


def test_noise_estimate_recovers_known_sigma():
    assert abs(estimate_noise_sigma(_fake_volume(sigma=2.0)) / 2.0 - 1) < 0.05


def test_stored_noise_sigma_matches_noise_in_slices(tmp_path):
    k = _fake_volume(sigma=1.0)
    esc = np.abs(_ifft2c(k))[:, 160:480, :]  # crop the readout oversampling -> 320 x 320
    with h5py.File(tmp_path / "file_x.h5", "w") as f:
        f["kspace"] = k
        f["reconstruction_esc"] = esc.astype(np.float32)
    preprocess_volume(tmp_path / "file_x.h5", tmp_path, size=128)
    z = np.load(tmp_path / "file_x.npz")
    # inside the flat block, the only variation left is noise
    flat = z["slices"][:, 30:70, 40:90]
    measured = np.mean([s.std() for s in flat])
    assert abs(float(z["noise_sigma"]) / measured - 1) < 0.1
