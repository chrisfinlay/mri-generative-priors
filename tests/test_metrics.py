"""Tests for metrics: PSNR/NMSE are TODO (Team C); SSIM/diversity are given."""

import numpy as np
from conftest import skip_if_todo

from mrigen import metrics


@skip_if_todo
def test_psnr_perfect_is_infinite_or_huge():
    x = np.random.default_rng(0).random((32, 32))
    assert metrics.psnr(x, x.copy(), data_range=1.0) > 80


@skip_if_todo
def test_nmse_zero_for_perfect():
    x = np.random.default_rng(1).random((32, 32))
    assert metrics.nmse(x, x.copy()) < 1e-12


def test_ssim_perfect_is_one():
    x = np.random.default_rng(2).random((32, 32))
    assert abs(metrics.ssim(x, x.copy()) - 1.0) < 1e-6


def test_diversity_zero_for_identical():
    x = np.random.default_rng(3).random((32, 32))
    stack = np.stack([x, x, x])
    assert metrics.diversity(stack) < 1e-6
