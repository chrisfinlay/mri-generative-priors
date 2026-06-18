"""Decoder-purity tests (GIVEN; CLAUDE.md Contract 4).

These do not depend on any TODO (only the given decoder + make_decoder_fn), so
they run on every branch.
"""

import jax
import jax.numpy as jnp

from mrigen.models.vae import VAE, make_decoder_fn

LATENT = 128  # CLAUDE.md Contract 4: latent_dim in 128-256


def _decode():
    vae = VAE(latent_dim=LATENT, key=jax.random.PRNGKey(0))
    return make_decoder_fn(vae)


def test_decoder_is_jittable_pure_fn():
    decode = _decode()
    z = jnp.zeros(LATENT)
    img = jax.jit(decode)(z)
    assert img.shape == (128, 128)
    assert jnp.all(jnp.isfinite(img))
    # purity: same input -> same output
    assert jnp.allclose(jax.jit(decode)(z), decode(z))


def test_decoder_is_gradable_through_z():
    decode = _decode()
    z = jnp.zeros(LATENT)
    g = jax.grad(lambda z: jnp.sum(decode(z) ** 2))(z)
    assert g.shape == (LATENT,)
    assert jnp.all(jnp.isfinite(g))


def test_make_decoder_fn_accepts_model_or_decoder():
    # Contract says make_decoder_fn(model); a bare Decoder is also accepted.
    vae = VAE(latent_dim=LATENT, key=jax.random.PRNGKey(0))
    z = jnp.zeros(LATENT)
    assert jnp.allclose(make_decoder_fn(vae)(z), make_decoder_fn(vae.decoder)(z))
