"""VAE-prior reconstruction in NumPyro: MAP (SVI) and posterior (NUTS).

Skeleton GIVEN; the ``recon_model`` body is the TODO (Team B). This is the
worked example from the probabilistic-programming lecture: put a standard normal
prior on the latent z, push it through the frozen decoder to get an image, apply
the forward operator, and place a Gaussian likelihood on the *observed* k-space
samples. Inference then turns measured k-space into a posterior over images.

Equinox detail (GIVEN): we ``eqx.partition`` the decoder into arrays + static
structure and recombine inside the model, so ``decoder(z)`` is a pure function
that JIT/NUTS can trace.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide

from mrigen.fourier import fft2c


def freeze_decoder(decoder):
    """Split a trained Equinox decoder into (params, static). GIVEN.

    Returns a pure callable ``decode(z) -> image`` safe for JIT/NUTS.
    """
    params, static = eqx.partition(decoder, eqx.is_array)

    def decode(z):
        dec = eqx.combine(params, static)
        return dec(z)

    return decode


def recon_model(y_obs, mask, decode, latent_dim, sigma):
    """NumPyro model: z ~ N(0, I); x = decode(z); Gaussian likelihood on k-space.

    TODO (Team B): implement the four lines
        1) sample ``z`` from a standard Normal of size ``latent_dim``;
        2) decode it to an image ``x``;
        3) form the forward measurement ``k = mask * fft2c(x)``;
        4) observe the real and imaginary parts of the sampled k-space with a
           Normal(., sigma) likelihood, conditioned on ``y_obs`` at the sampled
           locations (use ``mask.astype(bool)``).

    Reference (reveal if stuck):
        z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), 1.0))
        x = decode(z)
        k = mask * fft2c(x)
        obs = mask.astype(bool)
        numpyro.sample("y_re", dist.Normal(k.real[obs], sigma), obs=y_obs.real[obs])
        numpyro.sample("y_im", dist.Normal(k.imag[obs], sigma), obs=y_obs.imag[obs])
    """
    raise NotImplementedError("recon_model body is a TODO for Team B")


def reconstruct_map(
    y_obs, mask, decoder, latent_dim, sigma=0.01, *, steps=1000, lr=1e-2, seed=0
):
    """MAP reconstruction via SVI + AutoDelta (the Wednesday deliverable). GIVEN.

    Returns (image, z_map).
    """
    decode = freeze_decoder(decoder)
    guide = autoguide.AutoDelta(recon_model)
    svi = SVI(recon_model, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    result = svi.run(
        jax.random.PRNGKey(seed), steps, y_obs, mask, decode, latent_dim, sigma
    )
    z_map = result.params["z_auto_loc"]
    return decode(z_map), z_map


def reconstruct_posterior(
    y_obs,
    mask,
    decoder,
    latent_dim,
    sigma=0.01,
    *,
    num_samples=200,
    num_warmup=200,
    seed=0,
):
    """Posterior reconstruction via NUTS over z, with pixel-wise UQ. GIVEN.

    Keep ``latent_dim`` around 128-256 so the sampler mixes. Returns a dict with
    ``mean`` and ``std`` images (the std map is the uncertainty) and the raw
    image ``samples``.
    """
    decode = freeze_decoder(decoder)
    kernel = NUTS(recon_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed), y_obs, mask, decode, latent_dim, sigma)
    zs = mcmc.get_samples()["z"]
    images = jax.vmap(decode)(zs)
    return {
        "mean": jnp.mean(images, axis=0),
        "std": jnp.std(images, axis=0),
        "samples": images,
        "mcmc": mcmc,
    }
