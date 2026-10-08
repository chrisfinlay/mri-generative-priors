"""beta-VAE training loop with checkpointing.

GIVEN. Students choose hyperparameters (beta, latent_dim, epochs, lr) but the
loop, batching, and checkpoint I/O are provided. If training is slow, skip this
and load the pre-trained ``checkpoints/vae_128.eqx`` instead (see CHECKPOINTS.md).

Usage:
    python -m mrigen.train_vae --data data/processed --epochs 50 --beta 1.0

Trains on ``split="train"`` by default, i.e. without the held-out volumes in
``mrigen.data.HELDOUT_VOLUMES`` -- so that ``FastMRISlices(split="test")`` is a
fair evaluation set for the resulting checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from mrigen.data import FastMRISlices, data_loader
from mrigen.models.vae import VAE, vae_loss


def save_model(path: str | Path, model: VAE) -> None:
    """Serialise an Equinox model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(path), model)


def load_model(path: str | Path, latent_dim: int = 128) -> VAE:
    """Load an Equinox VAE; rebuilds the skeleton then fills in saved leaves."""
    skeleton = VAE(latent_dim=latent_dim, key=jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(str(path), skeleton)


@eqx.filter_jit
def _train_step(model, opt_state, batch, key, optim, beta):
    def batched_loss(m):
        keys = jax.random.split(key, batch.shape[0])
        losses, aux = jax.vmap(lambda x, k: vae_loss(m, x, k, beta))(batch, keys)
        recon, kl = aux
        return jnp.mean(losses), (jnp.mean(recon), jnp.mean(kl))

    (loss, aux), grads = eqx.filter_value_and_grad(batched_loss, has_aux=True)(model)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux


@eqx.filter_jit
def _eval_batch(model, batch, key, beta):
    keys = jax.random.split(key, batch.shape[0])
    losses, (recon, kl) = jax.vmap(lambda x, k: vae_loss(model, x, k, beta))(batch, keys)
    return jnp.sum(losses), jnp.sum(recon), jnp.sum(kl)


def evaluate_loss(model, slices, beta: float, *, batch_size: int = 256, seed: int = 0):
    """Mean (loss, recon, kl) over ``slices``, with a fixed key so epochs compare."""
    key = jax.random.PRNGKey(seed)
    total = jnp.zeros(3)
    for start in range(0, len(slices), batch_size):
        key, sk = jax.random.split(key)
        total += jnp.stack(_eval_batch(model, jnp.asarray(slices[start : start + batch_size]),
                                       sk, beta))
    return [float(t) / len(slices) for t in total]


def train(
    data_dir: str,
    *,
    latent_dim: int = 128,
    beta: float = 1.0,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    out: str = "checkpoints/vae_128.eqx",
    split: str | None = "train",
    val_split: str | None = "val",
) -> VAE:
    """Train the beta-VAE prior. ``split="train"`` keeps the held-out volumes out.

    After every epoch the train and ``val_split`` losses are appended to
    ``<out stem>.history.json`` next to the checkpoint, so the loss curves travel
    with the weights (and survive a run that is killed part-way).
    ``val_split=None`` skips the validation pass.
    """
    key = jax.random.PRNGKey(seed)
    model_key, key = jax.random.split(key)
    model = VAE(latent_dim=latent_dim, key=model_key)

    dataset = FastMRISlices(data_dir, split=split)
    print(f"training on {len(dataset)} slices from {len(dataset.volumes)} volume(s) "
          f"(split={split!r})")
    val = FastMRISlices(data_dir, split=val_split) if val_split else None
    history_path = Path(out).with_suffix(".history.json")
    history = {"config": {"latent_dim": latent_dim, "beta": beta, "batch_size": batch_size,
                          "lr": lr, "seed": seed, "split": split, "val_split": val_split},
               "epochs": []}
    # Clip the gradient norm so one bad batch can't throw the weights to NaN.
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    for epoch in range(epochs):
        ep_loss = ep_recon = ep_kl = 0.0
        n = 0
        for batch in data_loader(dataset, batch_size, seed=seed + epoch):
            key, sk = jax.random.split(key)
            batch = jnp.asarray(batch)
            model, opt_state, loss, (recon, kl) = _train_step(
                model, opt_state, batch, sk, optim, beta
            )
            ep_loss += float(loss)
            ep_recon += float(recon)
            ep_kl += float(kl)
            n += 1
        row = {"epoch": epoch, "train_loss": ep_loss / n, "train_recon": ep_recon / n,
               "train_kl": ep_kl / n}
        line = (f"epoch {epoch:3d}  loss {ep_loss / n:.4f}  "
                f"recon {ep_recon / n:.4f}  kl {ep_kl / n:.4f}")
        if val is not None:
            vl, vr, vk = evaluate_loss(model, val.slices, beta)
            row.update(val_loss=vl, val_recon=vr, val_kl=vk)
            line += f"  |  val loss {row['val_loss']:.4f}"
        print(line)
        history["epochs"].append(row)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=1))

    save_model(out, model)
    print(f"saved checkpoint -> {out}  (loss history -> {history_path})")
    return model


def main():
    p = argparse.ArgumentParser(description="Train the beta-VAE prior.")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="checkpoints/vae_128.eqx")
    p.add_argument("--split", default="train", help="'train' (default), 'val', 'test', or 'all'")
    p.add_argument("--val-split", default="val", help="split for per-epoch val loss, or 'none'")
    args = p.parse_args()
    train(
        args.data,
        latent_dim=args.latent_dim,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out=args.out,
        split=None if args.split == "all" else args.split,
        val_split=None if args.val_split == "none" else args.val_split,
    )


if __name__ == "__main__":
    main()
