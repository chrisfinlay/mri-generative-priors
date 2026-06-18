"""Normalisation round-trip tests (GIVEN; CLAUDE.md Contract 2)."""

import numpy as np

from mrigen.data import denormalise, normalise


def test_normalise_to_unit_range():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 7.5, size=(128, 128)).astype(np.float32)
    x_norm, scale = normalise(x)
    assert 0.0 <= x_norm.min() and x_norm.max() <= 1.0 + 1e-6
    assert abs(x_norm.max() - 1.0) < 1e-6  # max maps to 1
    assert np.isclose(scale, x.max(), rtol=1e-5)


def test_normalise_denormalise_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 3.0, size=(64, 64)).astype(np.float32)
    x_back = denormalise(*normalise(x))
    assert np.allclose(x_back, x, atol=1e-4)


def test_zero_image_is_safe():
    # All-zero slice must not divide by zero and must round-trip.
    x = np.zeros((16, 16), dtype=np.float32)
    x_norm, scale = normalise(x)
    assert scale == 1.0
    assert np.allclose(denormalise(x_norm, scale), x)


def test_scale_applied_identically_train_and_recon():
    # The scale captured at "train" time recovers the original at "recon" time,
    # so a reconstruction in [0, 1] maps back to the original intensities.
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 12.0, size=(32, 32)).astype(np.float32)
    x_norm, scale = normalise(x)            # train-time normalisation
    recon_norm = x_norm                     # a perfect [0, 1] reconstruction
    recon = denormalise(recon_norm, scale)  # recon-time, same scale
    assert np.allclose(recon, x, atol=1e-4)
