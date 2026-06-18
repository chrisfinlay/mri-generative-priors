"""Classical reconstruction baselines.

GIVEN: ``zero_filled`` -- the adjoint of the measurement, i.e. the naive
inverse FFT of the zero-filled k-space. This is the baseline every learned
method must beat.

TODO (optional): ``tv_fista`` -- a few FISTA iterations for TV / L1-wavelet
compressed sensing. Implement if time allows; otherwise the reference
implementation lives on the ``solutions`` branch.
"""

from __future__ import annotations

import jax.numpy as jnp

from mrigen.fourier import ifft2c


def zero_filled(y_obs: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Zero-filled reconstruction: real part of ifft2c of the masked k-space. GIVEN."""
    return ifft2c(mask * y_obs).real


def tv_fista(
    y_obs: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    lam: float = 1e-2,
    n_iter: int = 50,
    step: float = 1.0,
) -> jnp.ndarray:
    """TV-regularised CS reconstruction via FISTA.

    TODO (optional): minimise 1/2 ||M F x - y||^2 + lam * TV(x) with a few
    FISTA iterations (gradient step through the forward operator + soft-threshold
    of finite differences). Reference on the solutions branch.
    """
    raise NotImplementedError("tv_fista is an optional TODO")
