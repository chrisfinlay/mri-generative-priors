"""Normalisation round-trip tests (GIVEN; CLAUDE.md Contract 2)."""

import numpy as np
import pytest

from mrigen.data import FastMRISlices, denormalise, normalise, normalise_volume


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


def _shards(tmp_path):
    rng = np.random.default_rng(0)
    for name, n in (("file_a", 3), ("file_b", 2), ("file_c", 4)):
        slices = rng.random((n, 8, 8)).astype(np.float32)
        np.savez(tmp_path / f"{name}.npz", slices=slices, noise_sigma=np.float32(0.01))
    return tmp_path


def test_split_holds_out_named_volumes(tmp_path):
    root = _shards(tmp_path)
    everything = FastMRISlices(root)
    train = FastMRISlices(root, split="train", heldout=("file_b",))
    test = FastMRISlices(root, split="test", heldout=("file_b",))
    assert len(everything) == 9 and len(train) == 7 and len(test) == 2
    assert train.volumes == ["file_a", "file_c"] and test.volumes == ["file_b"]
    # laptop layout: "val" is the same held-out volumes as "test"
    assert FastMRISlices(root, split="val", heldout=("file_b",)).volumes == ["file_b"]
    assert list(train.volume_index) == [0, 0, 0, 1, 1, 1, 1]
    # train and test are disjoint
    assert not np.any(np.all(train.slices[:, None] == test.slices[None], axis=(2, 3)))


def test_split_errors_are_helpful(tmp_path):
    root = _shards(tmp_path)
    with pytest.raises(ValueError):
        FastMRISlices(root, split="validation")
    with pytest.raises(FileNotFoundError, match="Held-out volumes"):
        FastMRISlices(root, split="test", heldout=("file_missing",))


def _split_dirs(tmp_path):
    rng = np.random.default_rng(0)
    layout = {"train": ("file_a", "file_c"), "val": ("file_d",), "test": ("file_b",)}
    for split, names in layout.items():
        (tmp_path / split).mkdir()
        for name in names:
            slices = rng.random((2, 8, 8)).astype(np.float32)
            np.savez(tmp_path / split / f"{name}.npz", slices=slices, noise_sigma=np.float32(0.01))
    return tmp_path


def test_server_layout_picks_split_directory(tmp_path):
    root = _split_dirs(tmp_path)
    assert FastMRISlices(root, split="train").volumes == ["file_a", "file_c"]
    assert FastMRISlices(root, split="val").volumes == ["file_d"]
    # heldout= is ignored: the directory decides
    assert FastMRISlices(root, split="test", heldout=("file_a",)).volumes == ["file_b"]


def test_server_layout_refuses_mixed_splits(tmp_path):
    root = _split_dirs(tmp_path)
    with pytest.raises(ValueError, match="split must be one of"):
        FastMRISlices(root)


def test_volume_normalisation_shares_scale_and_noise(tmp_path):
    # Two slices of one volume: a bright one and a dim (edge-like) one.
    vol = np.stack([np.full((8, 8), 10.0), np.full((8, 8), 1.0)]).astype(np.float32)
    vol[0, 0, 0] = 1000.0  # one hot pixel must not set the scale
    np.savez(tmp_path / "file_v.npz", slices=vol, noise_sigma=np.float32(0.5))
    ds = FastMRISlices(tmp_path)
    assert ds.slices.min() >= 0.0 and ds.slices.max() <= 1.0
    assert ds.scales[0] == ds.scales[1]               # one scale per volume
    assert ds.slices[1].mean() < ds.slices[0].mean()  # the dim slice stays dim
    assert np.allclose(ds.sigmas, 0.5 / ds.scales)    # noise in normalised units
    assert np.allclose(denormalise(ds.slices[1], ds.scales[1]), vol[1], rtol=1e-5)


def test_normalise_volume_clips_hot_pixels():
    vol = np.ones((1, 40, 40), dtype=np.float32)  # 1600 pixels, one hot
    vol[0, 0, 0] = 100.0
    out, scale, sigma = normalise_volume(vol, noise_sigma=0.1)
    assert scale == 1.0 and out.max() == 1.0 and sigma == 0.1


def test_old_shards_without_noise_are_rejected(tmp_path):
    np.savez(tmp_path / "file_old.npz", slices=np.ones((2, 8, 8), np.float32))
    with pytest.raises(KeyError, match="noise_sigma"):
        FastMRISlices(tmp_path)
