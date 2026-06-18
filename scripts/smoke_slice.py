"""End-to-end vertical slice on SYNTHETIC data — no fastMRI required.

Builds a small dataset of smooth synthetic phantoms, trains a *tiny* VAE prior
(CPU-friendly, a few minutes), then reconstructs an undersampled phantom with
MAP (SVI + AutoDelta) and saves a 4-panel figure:

    ground truth | zero-filled | MAP | |error|

It is the runnable companion to ``tests/test_smoke.py`` (which asserts MAP beats
zero-filled). Everything obeys the single FFT convention in ``fourier.py`` and
the magnitude-path forward model in CLAUDE.md Contract 3.

    pixi run slice           # writes outputs/slice.png, prints PSNRs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrigen.data import normalise
from mrigen.masks import equispaced_mask
from mrigen.metrics import nmse, psnr
from mrigen.models.vae import VAE, make_decoder_fn
from mrigen.recon.classical import zero_filled
from mrigen.recon.operators import forward
from mrigen.recon.vae_numpyro import reconstruct_map

OUT = Path("outputs/slice.png")


def make_phantoms(n: int, size: int = 128, seed: int = 0) -> np.ndarray:
    """Smooth low-frequency phantoms (sums of 2D Gaussians) in [0, 1]."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    out = np.zeros((n, size, size), dtype=np.float32)
    for i in range(n):
        img = np.zeros((size, size), dtype=np.float32)
        for _ in range(rng.integers(3, 6)):
            cy, cx = rng.uniform(0.2, 0.8, size=2) * size
            w = rng.uniform(0.08, 0.22) * size
            amp = rng.uniform(0.5, 1.0)
            img += amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * w**2))
        out[i] = normalise(img)[0]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=128, help="synthetic phantoms")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--acceleration", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import equinox as eqx
    import optax

    from mrigen.data import data_loader
    from mrigen.models.vae import vae_loss

    key = jax.random.PRNGKey(args.seed)
    mk, key = jax.random.split(key)
    model = VAE(latent_dim=args.latent_dim, key=mk)

    data = make_phantoms(args.n, seed=args.seed)
    optim = optax.adam(1e-3)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, batch, k):
        def loss_fn(m):
            keys = jax.random.split(k, batch.shape[0])
            losses, _ = jax.vmap(lambda x, kk: vae_loss(m, x, kk, 1.0))(batch, keys)
            return jnp.mean(losses)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    print(f"training tiny VAE on {args.n} synthetic phantoms for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        ep, nb = 0.0, 0
        for batch in data_loader(data, batch_size=32, seed=args.seed + epoch):
            key, sk = jax.random.split(key)
            model, opt_state, loss = step(model, opt_state, jnp.asarray(batch), sk)
            ep += float(loss); nb += 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  loss {ep / nb:.4f}")

    # Ground truth: an image the *trained* prior represents (a sample from the
    # decoder). This is the regime where a learned prior helps — low-frequency
    # phantoms are recovered fine by zero-filling alone (the ACS band captures
    # them), so they don't show the prior's value. See the note in the report.
    decode = make_decoder_fn(model)
    x = decode(jax.random.normal(jax.random.PRNGKey(args.seed + 11), (args.latent_dim,)))
    mask = equispaced_mask((128, 128), args.acceleration)
    sigma = 0.01
    nk = jax.random.PRNGKey(args.seed + 99)
    noise = sigma * (jax.random.normal(nk, (128, 128)) + 1j * jax.random.normal(jax.random.PRNGKey(7), (128, 128)))
    y = forward(x, mask) + mask * noise

    x_zf = zero_filled(y, mask)
    x_map, _ = reconstruct_map(y, mask, model.decoder, args.latent_dim, sigma=sigma, steps=600, lr=2e-2)

    dr = float(x.max() - x.min())
    p_zf = psnr(np.asarray(x), np.asarray(x_zf), data_range=dr)
    p_map = psnr(np.asarray(x), np.asarray(x_map), data_range=dr)
    print(f"\nR={args.acceleration}  PSNR  zero-filled={p_zf:.2f} dB   MAP={p_map:.2f} dB")
    print(f"        NMSE  zero-filled={nmse(np.asarray(x), np.asarray(x_zf)):.4f}   "
          f"MAP={nmse(np.asarray(x), np.asarray(x_map)):.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    vmax = float(x.max())
    for a, img, title in (
        (ax[0], x, "ground truth"),
        (ax[1], x_zf, f"zero-filled ({p_zf:.1f} dB)"),
        (ax[2], x_map, f"MAP ({p_map:.1f} dB)"),
    ):
        a.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=vmax); a.set_title(title); a.axis("off")
    ax[3].imshow(np.abs(np.asarray(x_map) - np.asarray(x)), cmap="inferno")
    ax[3].set_title("|error| (MAP)"); ax[3].axis("off")
    fig.suptitle(f"Vertical slice on synthetic data — R={args.acceleration}")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"saved figure -> {OUT}")


if __name__ == "__main__":
    main()
