"""[STRETCH] Small UNet score model + VP-SDE utilities.

Optional stretch goal (see notebooks/05_diffusion_stretch.ipynb). Only attempt
after the VAE pipeline (train -> MAP -> posterior) works end to end. Provided as
a thin skeleton; teams that take this on flesh out the UNet blocks and the SDE
noise schedule.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class ScoreUNet(eqx.Module):
    """[STRETCH] Time-conditioned UNet predicting the score s_theta(x, t)."""

    def __init__(self, *, key):
        raise NotImplementedError("Diffusion is a stretch goal -- see SKILL plan section 9")

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:  # pragma: no cover
        raise NotImplementedError


def marginal_std(t: jnp.ndarray, sigma_min: float = 0.01, sigma_max: float = 50.0):
    """[STRETCH] VE-SDE marginal std at time t in [0, 1]."""
    return sigma_min * (sigma_max / sigma_min) ** t
