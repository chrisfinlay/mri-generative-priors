"""Tests for the forward operator and data consistency (TODO, Team B)."""

import jax.numpy as jnp
import numpy as np
from conftest import skip_if_todo

from mrigen.recon import operators as op


def _toy():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((16, 16)))
    mask = jnp.zeros((16, 16)).at[:, ::2].set(1.0)
    return x, mask


@skip_if_todo
def test_forward_shape_and_masking():
    x, mask = _toy()
    k = op.forward(x, mask)
    assert k.shape == x.shape
    # masked-out k-space entries are exactly zero
    assert jnp.allclose(k[mask == 0], 0.0)


@skip_if_todo
def test_data_consistency_matches_measurements():
    # If x_est == truth, DC projection returns the truth.
    x, mask = _toy()
    from mrigen.fourier import fft2c

    y = mask * fft2c(x)
    x_dc = op.data_consistency(x, y, mask)
    assert jnp.allclose(x_dc, x, atol=1e-4)
