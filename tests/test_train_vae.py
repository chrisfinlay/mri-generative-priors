"""Training loop plumbing (GIVEN): the loss history is written next to the checkpoint."""

import json

import numpy as np
from conftest import skip_if_unimplemented

from mrigen.train_vae import train


@skip_if_unimplemented
def test_train_writes_loss_history(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 2), ("val", 1)):
        (tmp_path / split).mkdir()
        for i in range(n):
            slices = rng.random((4, 128, 128)).astype(np.float32)
            np.savez(tmp_path / split / f"file_{split}{i}.npz", slices=slices,
                     noise_sigma=np.float32(0.02))
    out = tmp_path / "ckpt" / "vae.eqx"
    train(str(tmp_path), epochs=2, batch_size=4, out=str(out), split="train", val_split="val")

    history = json.loads((tmp_path / "ckpt" / "vae.history.json").read_text())
    assert out.exists()
    assert history["config"]["val_split"] == "val"
    assert [r["epoch"] for r in history["epochs"]] == [0, 1]
    for row in history["epochs"]:
        for k in ("train_loss", "train_recon", "train_kl", "val_loss", "val_recon", "val_kl"):
            assert np.isfinite(row[k])
