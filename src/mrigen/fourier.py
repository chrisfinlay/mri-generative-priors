"""Centred, orthonormal 2D Fourier transforms.

GIVEN. Do not modify the FFT convention. Using *one* convention everywhere
(centred + orthonormal "ortho" norm) is the single biggest source of silent
bugs in MRI reconstruction code: a half-pixel shift or a stray scale factor
produces images that look plausible but score badly. Everything downstream
(masks, forward operator, NumPyro likelihood) assumes these two functions.

Convention:
    fft2c(x)  = fftshift( fft2( ifftshift(x), norm="ortho") )
    ifft2c(k) = fftshift(ifft2( ifftshift(k), norm="ortho") )

so that the DC component sits at the centre of the array and
``ifft2c(fft2c(x)) == x`` (up to floating point error).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.numpy import fft


def fft2c(x: jnp.ndarray) -> jnp.ndarray:
    """Centred orthonormal 2D FFT over the last two axes (image -> k-space)."""
    x = jnp.asarray(x)
    return fft.fftshift(
        fft.fft2(fft.ifftshift(x, axes=(-2, -1)), norm="ortho", axes=(-2, -1)),
        axes=(-2, -1),
    )


def ifft2c(k: jnp.ndarray) -> jnp.ndarray:
    """Centred orthonormal 2D inverse FFT over the last two axes (k-space -> image)."""
    k = jnp.asarray(k)
    return fft.fftshift(
        fft.ifft2(fft.ifftshift(k, axes=(-2, -1)), norm="ortho", axes=(-2, -1)),
        axes=(-2, -1),
    )
