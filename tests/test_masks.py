"""Shape / acceleration tests for Cartesian masks (TODO, Team B)."""

import jax.numpy as jnp
import numpy as np

from mrigen.masks import acs_columns, equispaced_mask, random_mask
from conftest import skip_if_todo, skip_if_unimplemented


@skip_if_todo
def test_equispaced_shape_and_binary():
    m = equispaced_mask((128, 128), acceleration=4)
    assert m.shape == (128, 128)
    assert jnp.all((m == 0) | (m == 1))


@skip_if_todo
def test_equispaced_columns_consistent():
    # Whole columns are on or off: every row identical.
    m = equispaced_mask((128, 128), acceleration=4)
    assert jnp.all(m == m[0][None, :])


@skip_if_todo
def test_random_acceleration_approx():
    m = random_mask((128, 128), acceleration=4, seed=0)
    frac = float(m.mean())
    assert 0.15 < frac < 0.35  # ~1/4 of columns kept


@skip_if_todo
def test_acs_band_present():
    # Central columns must be fully sampled in both mask types.
    for m in (equispaced_mask((128, 128), 8), random_mask((128, 128), 8)):
        assert float(m[:, 60:68].mean()) == 1.0


@skip_if_unimplemented
def test_effective_acceleration_and_full_acs():
    # CLAUDE.md Contract 2: R = M.size / M.sum(). random_mask targets the overall
    # fraction so R_eff ≈ R; both mask types keep the ACS band fully sampled.
    R = 4
    acs = acs_columns(128, 0.08)
    for m in (equispaced_mask((128, 128), R), random_mask((128, 128), R)):
        m = np.asarray(m)
        r_eff = m.size / m.sum()
        assert 3.0 <= r_eff <= 4.5          # roughly 1/R kept (ACS makes it denser)
        assert float(m[:, acs].mean()) == 1.0  # ACS fully sampled
    # the randomised mask hits the target acceleration tightly
    assert abs(float(np.asarray(random_mask((128, 128), R)).mean()) - 1 / R) < 0.05
