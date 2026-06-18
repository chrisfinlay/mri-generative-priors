"""Cartesian undersampling masks.

TODO (Team B). This is where you internalise what "accelerating the scan"
means: a Cartesian MRI scan acquires k-space one phase-encode line (one
column) at a time, so we accelerate by *skipping columns*. We always keep a
fully-sampled block of central columns -- the auto-calibration signal (ACS) --
because the low frequencies carry most of the image energy and contrast.

You must implement:
    * ``equispaced_mask`` -- keep every R-th column plus an ACS band.
    * ``random_mask``     -- keep a random subset of columns plus an ACS band,
                             choosing the count so the *overall* acceleration is
                             approximately R.

Both return a real {0., 1.} array broadcastable over the k-space image, i.e.
shape ``(H, W)`` where whole columns are on or off.

Reference (give to students only if stuck): the ACS band is the central
``acs_frac * W`` columns; the effective acceleration counts the ACS columns as
already sampled.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def acs_columns(width: int, acs_frac: float) -> np.ndarray:
    """Indices of the central ACS columns (GIVEN helper).

    Returns the integer column indices of the fully-sampled centre band.
    """
    n_acs = max(1, int(round(acs_frac * width)))
    start = (width - n_acs) // 2
    return np.arange(start, start + n_acs)


def equispaced_mask(
    shape: tuple[int, int],
    acceleration: int,
    acs_frac: float = 0.08,
) -> jnp.ndarray:
    """Equispaced Cartesian mask with a central ACS band.

    Args:
        shape: (H, W) of the k-space image.
        acceleration: R, keep roughly every R-th phase-encode column.
        acs_frac: fraction of columns kept fully-sampled in the centre.

    Returns:
        (H, W) float array of 0./1.; entire columns are on or off.
    """
    # SOLUTION
    h, w = shape
    cols = np.zeros(w, dtype=np.float32)
    cols[::acceleration] = 1.0
    cols[acs_columns(w, acs_frac)] = 1.0
    return jnp.asarray(np.broadcast_to(cols[None, :], (h, w)).copy())


def random_mask(
    shape: tuple[int, int],
    acceleration: int,
    acs_frac: float = 0.08,
    seed: int = 0,
) -> jnp.ndarray:
    """Random Cartesian mask with a central ACS band.

    Args:
        shape: (H, W) of the k-space image.
        acceleration: target overall R.
        acs_frac: fraction of columns kept fully-sampled in the centre.
        seed: RNG seed for reproducibility.

    Returns:
        (H, W) float array of 0./1.; entire columns are on or off.
    """
    # SOLUTION
    h, w = shape
    rng = np.random.default_rng(seed)
    acs = acs_columns(w, acs_frac)
    cols = np.zeros(w, dtype=np.float32)
    cols[acs] = 1.0
    n_target = int(round(w / acceleration))
    n_extra = max(0, n_target - len(acs))
    candidates = np.setdiff1d(np.arange(w), acs)
    chosen = rng.choice(candidates, size=min(n_extra, len(candidates)), replace=False)
    cols[chosen] = 1.0
    return jnp.asarray(np.broadcast_to(cols[None, :], (h, w)).copy())
