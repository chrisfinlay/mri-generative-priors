"""Pre-flight check: print JAX devices and run a 1-step VAE forward.

GIVEN. Run this before lunch on Monday so a group knows the GPU works:

    pixi run check

It does NOT need any data -- it makes a random 128x128 image and pushes it
through a freshly-initialised VAE. If the reparameterisation TODO is still
unimplemented it will tell you so explicitly (that's expected on day one).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def main() -> None:
    print("JAX version:", jax.__version__)
    print("JAX devices:", jax.devices())

    from mrigen.models.vae import VAE, vae_loss

    key = jax.random.PRNGKey(0)
    mk, xk, lk = jax.random.split(key, 3)
    model = VAE(latent_dim=128, key=mk)
    x = jax.random.uniform(xk, (128, 128))

    mu, logvar = model.encoder(x)
    print(f"encoder OK: mu {mu.shape}, logvar {logvar.shape}")

    try:
        loss, (recon, kl) = vae_loss(model, x, lk, beta=1.0)
        print(f"1-step VAE forward OK: loss={float(loss):.4f} recon={float(recon):.4f} kl={float(kl):.4f}")
    except NotImplementedError as e:
        print(f"VAE loss not runnable yet (expected on day 1): {e}")
        print("-> implement reparameterise() in src/mrigen/models/vae.py")

    # decoder is a pure function of z (what NumPyro needs)
    z = jnp.zeros(128)
    img = model.decoder(z)
    print(f"decoder OK: image {img.shape}, min {float(img.min()):.3f} max {float(img.max()):.3f}")
    print("setup check complete.")


if __name__ == "__main__":
    main()
