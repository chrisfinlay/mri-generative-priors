"""Plotting helpers for presentation-ready figures.

GIVEN. Consistent colour maps and layouts so every team's slides look the same:
images in greyscale, k-space log-magnitude, error maps in a perceptual map, and
uncertainty (std) maps in 'magma'. All functions take a matplotlib Axes (or
create a figure) and never call plt.show() so they compose in notebooks.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def show_image(x, ax=None, title=None, cmap="gray", vmin=None, vmax=None):
    """Show a magnitude image."""
    ax = ax or plt.gca()
    im = ax.imshow(np.asarray(x), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title or "")
    ax.axis("off")
    return im


def show_kspace(k, ax=None, title="k-space (log-mag)"):
    """Show log-magnitude of complex k-space."""
    ax = ax or plt.gca()
    mag = np.log1p(np.abs(np.asarray(k)))
    im = ax.imshow(mag, cmap="viridis")
    ax.set_title(title)
    ax.axis("off")
    return im


def show_error(gt, pred, ax=None, title="|error|", cmap="inferno"):
    """Show absolute error map."""
    ax = ax or plt.gca()
    err = np.abs(np.asarray(pred) - np.asarray(gt))
    im = ax.imshow(err, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    return im


def show_uncertainty(std, ax=None, title="uncertainty (std)", cmap="magma"):
    """Show pixel-wise posterior std."""
    ax = ax or plt.gca()
    im = ax.imshow(np.asarray(std), cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    return im


def panel(gt, pred, std=None, mask=None):
    """One-row summary panel: ground truth | recon | error | (uncertainty).

    Returns the matplotlib Figure so notebooks can savefig for slides.
    """
    cols = 3 + (std is not None)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    vmax = float(np.asarray(gt).max())
    show_image(gt, axes[0], "ground truth", vmin=0, vmax=vmax)
    show_image(pred, axes[1], "reconstruction", vmin=0, vmax=vmax)
    show_error(gt, pred, axes[2])
    if std is not None:
        show_uncertainty(std, axes[3])
    fig.tight_layout()
    return fig
