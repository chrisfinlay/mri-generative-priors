"""Round-trip and convention tests for the centred FFT (GIVEN)."""

import jax.numpy as jnp
import numpy as np

from mrigen.fourier import fft2c, ifft2c


def test_roundtrip():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((32, 32)))
    back = ifft2c(fft2c(x))
    assert jnp.allclose(back.real, x, atol=1e-5)
    assert jnp.allclose(back.imag, 0.0, atol=1e-5)


def test_orthonormal_energy():
    # Parseval: an orthonormal transform preserves L2 energy.
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((16, 16)))
    assert jnp.allclose(jnp.sum(jnp.abs(fft2c(x)) ** 2), jnp.sum(x**2), atol=1e-4)


def test_dc_is_centred():
    # A constant image puts all energy at the centre (DC) pixel.
    x = jnp.ones((8, 8))
    k = fft2c(x)
    peak = jnp.unravel_index(jnp.argmax(jnp.abs(k)), k.shape)
    assert tuple(int(i) for i in peak) == (4, 4)
