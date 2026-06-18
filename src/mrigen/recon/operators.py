"""Forward operator, adjoint, and data-consistency projection.

TODO (Team B). Short, central, conceptual -- this is the heart of the inverse
problem. The measurement model is

    A(x) = M . F(x)          (undersampled k-space of image x)

where F = fourier.fft2c (centred orthonormal FFT) and M is the binary Cartesian
mask from masks.py. The adjoint maps k-space back to image space, and the
data-consistency (DC) projection forces the reconstruction to agree with the
*measured* k-space samples while keeping the model's estimate everywhere else.
"""

from __future__ import annotations

import jax.numpy as jnp

from mrigen.fourier import fft2c, ifft2c


def forward(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """A(x) = mask * fft2c(x): image -> undersampled k-space.

    TODO (Team B): return the masked centred FFT of x.
    """
    # SOLUTION
    return mask * fft2c(x)


def adjoint(k: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """A^H(k) = ifft2c(mask * k): undersampled k-space -> image.

    TODO (Team B): apply the mask then the centred inverse FFT. (For a real
    magnitude image you will usually take ``.real`` downstream.)
    """
    # SOLUTION
    return ifft2c(mask * k)


def data_consistency(x_est: jnp.ndarray, y_obs: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Replace estimated k-space with measured samples where the mask is on.

    Returns the image
        ifft2c( mask * y_obs + (1 - mask) * fft2c(x_est) )
    i.e. keep the measured entries, trust the model elsewhere.

    TODO (Team B): implement the DC projection above and return the (real)
    image.
    """
    # SOLUTION
    k_est = fft2c(x_est)
    k_dc = mask * y_obs + (1.0 - mask) * k_est
    return ifft2c(k_dc).real
