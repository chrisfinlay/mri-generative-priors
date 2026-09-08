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


def test_plot_handles_partial_acceleration_series():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = {"slice": "0", "R_eff": 3.3, "psnr": 30.0, "ssim": 0.9, "nmse": 0.01,
           "seconds": 0.1, "has_std": False}
    rows = [
        {**row, "method": "a", "R": 4}, {**row, "method": "a", "R": 8},
        {**row, "method": "b", "R": 8, "psnr": 25.0},   # b has no R=4 row
    ]
    fig, ax = plt.subplots()
    ev.plot_metric_vs_R(rows, "psnr", ax=ax)
    xdata = [tuple(np.asarray(ln.get_xdata()).tolist())
             for ln in ax.get_lines() if len(ln.get_xdata())]
    assert (8.0,) in xdata, "a method with only R=8 must be plotted at R=8, not shifted to R=4"
    plt.close(fig)


def test_sweep_machinery_runs_even_while_metrics_are_todos(monkeypatch):
    # Exercise rows/pooling/worst-case/timing on the student branch too, by standing in
    # for the psnr/nmse TODOs. The real-metrics path is covered by the test above.
    monkeypatch.setattr(ev.metrics, "psnr",
                        lambda gt, pred, data_range=1.0: -float(np.mean((gt - pred) ** 2)))
    monkeypatch.setattr(ev.metrics, "nmse", lambda gt, pred: float(np.mean((gt - pred) ** 2)))
    imgs = _images(2)
    res = ev.evaluate({"zero-filled": ev.zero_filled_recon}, imgs, (2,), sigma=0.01,
                      mask_fn=_mask_fn, verbose=False)
    assert len(res.rows) == 2 and all(r["seconds"] >= 0 for r in res.rows)
    assert res.worst["zero-filled"]["slice"] in ("0", "1")
    assert "| zero-filled |" in ev.table(res.rows)


def test_max_tree_depth_reaches_nuts(monkeypatch):
    from mrigen.recon import vae_numpyro as vn

    captured = {}

    class FakeNUTS:
        def __init__(self, model, max_tree_depth=10, **kw):
            captured["depth"] = max_tree_depth

    class FakeMCMC:
        def __init__(self, kernel, **kw):
            pass

        def run(self, *a, **k):
            pass

        def get_samples(self):
            return {"z": jnp.zeros((3, 4))}

    monkeypatch.setattr(vn, "NUTS", FakeNUTS)
    monkeypatch.setattr(vn, "MCMC", FakeMCMC)

    def decoder(z):
        return jnp.zeros((8, 8)) + z.sum()

    out = vn.reconstruct_posterior(jnp.zeros((8, 8), jnp.complex64), jnp.ones((8, 8)),
                                   decoder, 4, max_tree_depth=6)
    assert captured["depth"] == 6
    assert out["mean"].shape == (8, 8) and out["std"].shape == (8, 8)


def test_evaluate_rejects_bad_inputs():
    import pytest

    with pytest.raises(ValueError):
        ev.evaluate({"zf": ev.zero_filled_recon}, np.zeros((0, 8, 8), np.float32), (2,),
                    mask_fn=_mask_fn, verbose=False)
    with pytest.raises(ValueError):
        ev.measure(jnp.zeros((8, 8)), jnp.ones((8, 8)), -0.5, jax.random.PRNGKey(0))
