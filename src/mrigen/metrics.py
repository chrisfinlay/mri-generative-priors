"""Reconstruction metrics.

Mixed given / TODO.

TODO (Team C): ``psnr`` and ``nmse`` -- two-line formulas, build confidence and
force you to think about what "reconstruction error" means.

GIVEN: ``ssim`` (delegates to scikit-image), ``diversity`` (mean pairwise SSIM
over posterior samples -- low diversity means the prior is too confident), and
``calibration_curve`` (bins predicted std against actual error -- a well
calibrated uncertainty has error growing with predicted std).
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def psnr(gt: np.ndarray, pred: np.ndarray, data_range: float | None = None) -> float:
    """Peak signal-to-noise ratio in dB (higher is better).

    PSNR = 10 * log10(data_range**2 / MSE).
    """
    # SOLUTION
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if data_range is None:
        data_range = float(gt.max() - gt.min())
    mse = float(np.mean((gt - pred) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(data_range**2 / mse)


def nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Normalised mean squared error (lower is better).

    NMSE = ||pred - gt||^2 / ||gt||^2.
    """
    # SOLUTION
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.sum((pred - gt) ** 2) / np.sum(gt**2))


def ssim(gt: np.ndarray, pred: np.ndarray, data_range: float | None = None) -> float:
    """Structural similarity index (higher is better). GIVEN."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if data_range is None:
        data_range = float(gt.max() - gt.min())
    return float(structural_similarity(gt, pred, data_range=data_range))


def diversity(samples: np.ndarray) -> float:
    """Mean pairwise (1 - SSIM) over a stack of posterior image samples. GIVEN.

    Args:
        samples: (N, H, W) array of decoded posterior samples.

    Returns:
        Mean pairwise dissimilarity in [0, 1]; 0 means identical samples.
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = samples.shape[0]
    if n < 2:
        return 0.0
    dr = float(samples.max() - samples.min())
    diss = []
    for i in range(n):
        for j in range(i + 1, n):
            diss.append(1.0 - structural_similarity(samples[i], samples[j], data_range=dr))
    return float(np.mean(diss))


def calibration_curve(
    error: np.ndarray, std: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Bin absolute error against predicted std. GIVEN.

    Args:
        error: |mean - ground_truth|, any shape.
        std: pixel-wise posterior std, same shape.
        n_bins: number of quantile bins over std.

    Returns:
        (mean_std_per_bin, mean_error_per_bin); a calibrated model has these
        two arrays roughly proportional.
    """
    error = np.asarray(error).ravel()
    std = np.asarray(std).ravel()
    order = np.argsort(std)
    std, error = std[order], error[order]
    edges = np.linspace(0, len(std), n_bins + 1).astype(int)
    mean_std = np.array([std[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    mean_err = np.array([error[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    return mean_std, mean_err
