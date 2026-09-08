"""Model-agnostic evaluation: one protocol for every reconstruction method. GIVEN.

Comparisons are only fair if every method sees the *same* problem. This module
fixes the problem and lets the method vary. A **reconstructor** is any callable

    recon(y_obs, mask, sigma) -> Recon

where :class:`Recon` carries ``mean`` (H, W) and, if the method has them,
``std`` (H, W) and ``samples`` (N, H, W). Adapters for the repo's own methods
are below; a new model -- a diffusion prior, a power-spectrum prior, a classical
solver, anything -- joins the comparison with an adapter of the same shape.

The protocol (what :func:`evaluate` standardises, and what your table must state
-- the held-out slices are yours to supply; it cannot check where an array came from):

* **held-out slices only** -- ``FastMRISlices(root, split="test")``, never
  slices the prior was trained on;
* the **same mask, noise draw and sigma for every method**, seeded per
  (slice, R), so differences are due to the method and nothing else;
* metrics against the fully-sampled ground truth with ``data_range=1``;
* the **effective** acceleration ``R_eff = M.size / M.sum()``, not the nominal R;
* **wall-clock seconds** per reconstruction, after one warm-up call so JIT
  compilation is not charged to the first slice;
* **calibration** for methods that return ``std``: pooled |error| against
  predicted std, in quantile bins (:func:`mrigen.metrics.calibration_curve`).

Numbers are per (method, slice, R) rows; :func:`summarise` gives mean +/- std
over slices, :func:`table` a markdown table, :func:`plot_metric_vs_R` the curve,
:func:`worst_case` the slice a method got most wrong -- show that one too.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from mrigen import metrics
from mrigen.fourier import fft2c
from mrigen.recon.classical import tv_fista, zero_filled
from mrigen.recon.spectrum import wiener_reconstruct
from mrigen.recon.vae_numpyro import reconstruct_map, reconstruct_posterior

# --------------------------------------------------------------------------- interface


@dataclass
class Recon:
    """What a reconstructor returns. ``mean`` is required; the rest is optional."""

    mean: np.ndarray
    std: np.ndarray | None = None
    samples: np.ndarray | None = None


def measure(x, mask, sigma: float, key) -> jnp.ndarray:
    """Simulate the measurement ``y = M * (fft2c(x) + noise)`` (CLAUDE.md Contract 3).

    Noise is complex Gaussian with std ``sigma`` per real/imag component, drawn from
    ``key`` -- so the same ``key`` gives every method the same measurement.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    x = jnp.asarray(x)
    k1, k2 = jax.random.split(key)
    noise = sigma * (jax.random.normal(k1, x.shape) + 1j * jax.random.normal(k2, x.shape))
    return jnp.asarray(mask) * (fft2c(x) + noise)


# --------------------------------------------------------------------------- adapters
# Each returns a reconstructor: (y_obs, mask, sigma) -> Recon. Copy the pattern.


def zero_filled_recon(y_obs, mask, sigma) -> Recon:
    """The baseline: inverse FFT with zeros in the gaps."""
    return Recon(mean=np.asarray(zero_filled(y_obs, mask)))


def tv_recon(lam: float = 1e-2, n_iter: int = 50):
    """Classical compressed sensing (FISTA with an L1 sparsity proxy; see classical.py)."""

    def recon(y_obs, mask, sigma) -> Recon:
        return Recon(mean=np.asarray(tv_fista(y_obs, mask, lam=lam, n_iter=n_iter)))

    return recon


def wiener_recon(P, n_samples: int = 0, seed: int = 0):
    """Power-spectrum prior with its closed-form posterior (recon/spectrum.py)."""

    def recon(y_obs, mask, sigma) -> Recon:
        out = wiener_reconstruct(
            y_obs, mask, P, sigma, n_samples=n_samples, key=jax.random.PRNGKey(seed)
        )
        samples = None if out["samples"] is None else np.asarray(out["samples"])
        return Recon(mean=np.asarray(out["mean"]), std=np.asarray(out["std"]), samples=samples)

    return recon


def map_recon(decoder, latent_dim, steps: int = 1000, lr: float = 1e-2, seed: int = 0):
    """MAP through *any* decoder -- a VAE module or a pure ``decode(z)`` function.

    ``latent_dim`` may be an int (VAE) or a shape tuple (e.g. ``(2, H, W)`` for the
    power-spectrum decoder). Returns no uncertainty: MAP is a point estimate.
    """

    def recon(y_obs, mask, sigma) -> Recon:
        x_map, _ = reconstruct_map(
            y_obs, mask, decoder, latent_dim, sigma=sigma, steps=steps, lr=lr, seed=seed
        )
        return Recon(mean=np.asarray(x_map))

    return recon


def posterior_recon(
    decoder, latent_dim, num_samples: int = 200, num_warmup: int = 200, seed: int = 0,
    max_tree_depth: int = 10,
):
    """NUTS posterior through a decoder: mean, per-pixel std, and the samples.

    NUTS is the expensive method: up to ``2**max_tree_depth - 1`` decoder evaluations
    per sample. On a CPU, use few samples and ``max_tree_depth`` 6-7; on the GPU
    server, the defaults.
    """

    def recon(y_obs, mask, sigma) -> Recon:
        out = reconstruct_posterior(
            y_obs, mask, decoder, latent_dim, sigma=sigma,
            num_samples=num_samples, num_warmup=num_warmup, seed=seed,
            max_tree_depth=max_tree_depth,
        )
        return Recon(
            mean=np.asarray(out["mean"]),
            std=np.asarray(out["std"]),
            samples=np.asarray(out["samples"]),
        )

    return recon


# --------------------------------------------------------------------------- the sweep


@dataclass
class Results:
    """Output of :func:`evaluate`.

    ``rows``: one dict per (method, slice, R) with the metrics.
    ``pooled``: method -> (|error| pixels, std pixels) over every evaluated slice,
    for calibration; only methods that return ``std`` appear.
    ``worst``: method -> the lowest-PSNR case (ground truth, recon, std, R, slice).
    """

    rows: list[dict] = field(default_factory=list)
    pooled: dict[str, tuple[list, list]] = field(
        default_factory=lambda: defaultdict(lambda: ([], []))
    )
    worst: dict[str, dict] = field(default_factory=dict)

    def methods(self) -> list[str]:
        seen = []
        for r in self.rows:
            if r["method"] not in seen:
                seen.append(r["method"])
        return seen

    def accelerations(self) -> list[int]:
        return sorted({r["R"] for r in self.rows})


def _block(x):
    """Wait for async JAX work so timings are honest."""
    try:
        jax.block_until_ready(x)
    except Exception:  # plain numpy or python objects
        pass


def evaluate(
    methods: dict,
    images,
    accelerations=(4, 8),
    *,
    sigma: float = 0.01,
    mask_fn=None,
    seed: int = 0,
    warmup: bool = True,
    labels=None,
    verbose: bool = True,
) -> Results:
    """Run every method on every (held-out) image at every acceleration.

    Args:
        methods: ``{"name": reconstructor}``; see the adapters above.
        images: ``(N, H, W)`` ground-truth magnitude images in [0, 1] -- use the
            *test* split.
        accelerations: nominal R values to sweep.
        sigma: measurement noise std, shared by every method.
        mask_fn: ``mask_fn((H, W), R) -> mask``; defaults to
            :func:`mrigen.masks.equispaced_mask`.
        seed: base seed; the measurement for (slice i, R) is seeded from it.
        warmup: call each method once untimed first (JIT compilation).
        labels: optional names for the images (e.g. "file1000593/12").
        verbose: print one line per (slice, R).
    """
    if mask_fn is None:
        from mrigen.masks import equispaced_mask

        mask_fn = equispaced_mask
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3 or len(images) == 0:
        raise ValueError(f"images must be a non-empty (N, H, W) stack, got shape {images.shape}")
    res = Results()
    warmed = set()
    for i, x in enumerate(images):
        label = labels[i] if labels is not None else str(i)
        for R in accelerations:
            mask = jnp.asarray(mask_fn(x.shape, R))
            r_eff = float(mask.size / mask.sum())
            key = jax.random.PRNGKey(seed * 100_003 + i * 1_009 + int(R))
            y = measure(x, mask, sigma, key)
            for name, fn in methods.items():
                if warmup and name not in warmed:
                    _block(fn(y, mask, sigma).mean)
                    warmed.add(name)
                t0 = time.perf_counter()
                out = fn(y, mask, sigma)
                _block(out.mean)
                seconds = time.perf_counter() - t0
                mean = np.asarray(out.mean, dtype=np.float32)
                row = {
                    "method": name,
                    "slice": label,
                    "R": int(R),
                    "R_eff": r_eff,
                    "psnr": float(metrics.psnr(x, mean, data_range=1.0)),
                    "ssim": float(metrics.ssim(x, mean, data_range=1.0)),
                    "nmse": float(metrics.nmse(x, mean)),
                    "seconds": seconds,
                    "has_std": out.std is not None,
                }
                res.rows.append(row)
                if out.std is not None:
                    errs, stds = res.pooled[name]
                    errs.append(np.abs(mean - x).ravel())
                    stds.append(np.asarray(out.std, dtype=np.float32).ravel())
                w = res.worst.get(name)
                if w is None or row["psnr"] < w["psnr"]:
                    res.worst[name] = {
                        "psnr": row["psnr"], "R": int(R), "slice": label, "gt": x, "mean": mean,
                        "std": None if out.std is None else np.asarray(out.std),
                    }
            if verbose:
                done = [r for r in res.rows if r["slice"] == label and r["R"] == R]
                print(
                    f"slice {label:>14}  R={R} (eff {r_eff:.1f})  "
                    + "  ".join(
                        f"{r['method']} {r['psnr']:.1f} dB ({r['seconds']:.0f}s)" for r in done
                    )
                )
    return res


# --------------------------------------------------------------------------- reporting


def summarise(rows, metric_names=("psnr", "ssim", "nmse", "seconds")) -> dict:
    """``{(method, R): {metric: (mean, std, n)}}`` over slices."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["R"])].append(r)
    out = {}
    for key, rs in groups.items():
        out[key] = {
            m: (float(np.mean([r[m] for r in rs])), float(np.std([r[m] for r in rs])), len(rs))
            for m in metric_names
        }
        out[key]["R_eff"] = float(np.mean([r["R_eff"] for r in rs]))
    return out


def table(rows, metric: str = "psnr", fmt: str = "{:.2f}") -> str:
    """Markdown table, methods x accelerations, ``mean +/- std`` over slices."""
    s = summarise(rows)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    accs = sorted({r["R"] for r in rows})
    r_eff = {R: np.mean([v["R_eff"] for k, v in s.items() if k[1] == R]) for R in accs}
    head = f"| {metric} | " + " | ".join(f"R = {R} (eff {r_eff[R]:.1f})" for R in accs) + " |"
    lines = [head, "|---|" + "---|" * len(accs)]
    for m in methods:
        cells = []
        for R in accs:
            v = s.get((m, R))
            if v is None:
                cells.append("—")
            else:
                cells.append(f"{fmt.format(v[metric][0])} ± {fmt.format(v[metric][1])}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    n = max(v[metric][2] for v in s.values())
    lines.append(f"\n(mean ± std over {n} held-out slice(s))")
    return "\n".join(lines)


def plot_metric_vs_R(rows, metric: str = "psnr", ax=None):
    """One line per method: ``metric`` against nominal R, error bars = std over slices."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    s = summarise(rows)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    accs = sorted({r["R"] for r in rows})
    for m in methods:
        present = [R for R in accs if (m, R) in s]
        mu = [s[(m, R)][metric][0] for R in present]
        sd = [s[(m, R)][metric][1] for R in present]
        ax.errorbar(present, mu, yerr=sd, marker="o", capsize=3, label=m)
    ax.set_xlabel("acceleration R (nominal)")
    ax.set_ylabel(metric)
    ax.set_xticks(accs)
    ax.legend()
    return ax


def calibration(results: Results, method: str, n_bins: int = 10):
    """Pooled calibration curve for one method: ``(mean_std_per_bin, mean_err_per_bin)``."""
    errs, stds = results.pooled[method]
    if not errs:
        raise ValueError(f"{method!r} returned no std, so it has no calibration curve")
    return metrics.calibration_curve(np.concatenate(errs), np.concatenate(stds), n_bins=n_bins)


def plot_calibration(results: Results, methods=None, n_bins: int = 10, ax=None):
    """Calibration curves for every method with a std, against the Gaussian reference.

    For a calibrated zero-mean Gaussian error the mean **absolute** error in a bin is
    ``sqrt(2/pi) ~ 0.8`` of the predicted std, so the reference line has that slope,
    not 1.
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    methods = methods or [m for m in results.methods() if results.pooled[m][0]]
    hi = 0.0
    for m in methods:
        s, e = calibration(results, m, n_bins)
        ax.plot(s, e, "o-", label=m)
        hi = max(hi, float(s.max()), float(e.max()))
    k = float(np.sqrt(2.0 / np.pi))
    ax.plot([0, hi], [0, k * hi], "k:", lw=1, label="perfect (√(2/π) · std)")
    ax.set_xlabel("predicted std (bin mean)")
    ax.set_ylabel("actual |error| (bin mean)")
    ax.legend()
    return ax


def worst_case(results: Results, method: str):
    """Figure of the slice ``method`` got most wrong: truth | recon | error | (std)."""
    from mrigen import viz

    w = results.worst[method]
    fig = viz.panel(w["gt"], w["mean"], std=w["std"])
    fig.suptitle(f"{method}: worst case, slice {w['slice']}, R = {w['R']}, {w['psnr']:.1f} dB")
    return fig
