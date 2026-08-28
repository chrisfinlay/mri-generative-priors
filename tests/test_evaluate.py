"""The model-agnostic evaluation protocol (GIVEN)."""

import jax
import jax.numpy as jnp
import numpy as np
from conftest import skip_if_unimplemented

from mrigen import evaluate as ev
from mrigen.fourier import fft2c


def _mask_fn(shape, R):
    """Plain numpy mask so these tests do not depend on the masks.py TODO."""
    h, w = shape
    m = np.zeros((h, w), np.float32)
    m[:, ::R] = 1.0
    m[:, w // 2 - 2 : w // 2 + 2] = 1.0
    return m


def _images(n=2, size=32, seed=0):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    out = np.zeros((n, size, size), np.float32)
    for i in range(n):
        for _ in range(3):
            cy, cx = rng.uniform(0.2, 0.8, 2) * size
            out[i] += np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * (0.15 * size) ** 2))
        out[i] /= out[i].max()
    return out


def test_measure_matches_contract():
    x = jnp.asarray(_images(1)[0])
    mask = jnp.asarray(_mask_fn(x.shape, 4))
    y = ev.measure(x, mask, 0.0, jax.random.PRNGKey(0))
    assert bool(jnp.allclose(y, mask * fft2c(x)))
    y1 = ev.measure(x, mask, 0.1, jax.random.PRNGKey(3))
    y2 = ev.measure(x, mask, 0.1, jax.random.PRNGKey(3))
    assert bool(jnp.allclose(y1, y2))  # same key -> same measurement for every method


def test_summarise_and_table_are_pure():
    def row(method, sl, psnr, ssim, nmse, seconds, has_std):
        return {"method": method, "slice": sl, "R": 4, "R_eff": 3.3, "psnr": psnr, "ssim": ssim,
                "nmse": nmse, "seconds": seconds, "has_std": has_std}

    rows = [
        row("a", "0", 30.0, 0.9, 0.01, 0.1, False),
        row("a", "1", 32.0, 0.92, 0.008, 0.1, False),
        row("b", "0", 20.0, 0.5, 0.1, 1.0, True),
        row("b", "1", 22.0, 0.55, 0.09, 1.0, True),
    ]
    s = ev.summarise(rows)
    assert s[("a", 4)]["psnr"] == (31.0, 1.0, 2)
    md = ev.table(rows, "psnr")
    assert "| a |" in md and "| b |" in md and "31.00 ± 1.00" in md and "eff 3.3" in md


@skip_if_unimplemented
def test_sweep_zero_filled_vs_zeros_with_calibration():
    # 'zeros' returns an empty image with a std map: it must lose on PSNR and show up in
    # the calibration pool; zero-filled has no std and must not.
    imgs = _images(2)

    def zeros(y, mask, sigma):
        shape = imgs.shape[1:]
        return ev.Recon(mean=np.zeros(shape, np.float32), std=np.full(shape, 0.3, np.float32))

    methods = {"zero-filled": ev.zero_filled_recon, "zeros": zeros}
    res = ev.evaluate(methods, imgs, (2, 4), sigma=0.01, mask_fn=_mask_fn, verbose=False)
    assert len(res.rows) == 2 * 2 * 2
    assert set(res.rows[0]) >= {"method", "slice", "R", "R_eff", "psnr", "ssim", "nmse", "seconds"}
    assert all(1.5 < r["R_eff"] <= r["R"] for r in res.rows)
    s = ev.summarise(res.rows)
    for R in (2, 4):
        assert s[("zero-filled", R)]["psnr"][0] > s[("zeros", R)]["psnr"][0]
    assert "zeros" in res.pooled and "zero-filled" not in res.pooled
    ms, me = ev.calibration(res, "zeros", n_bins=5)
    assert len(ms) <= 5 and np.allclose(ms, 0.3)
    assert res.worst["zero-filled"]["R"] == 4  # worst case is at the highest acceleration
    assert "| zero-filled |" in ev.table(res.rows)
