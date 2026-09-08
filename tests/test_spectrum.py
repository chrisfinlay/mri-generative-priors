"""Power-spectrum prior: spectrum estimate, decoder, closed-form posterior (GIVEN)."""

import jax
import jax.numpy as jnp
import numpy as np

from mrigen.fourier import fft2c
from mrigen.recon.classical import zero_filled
from mrigen.recon.spectrum import estimate_power_spectrum, make_spectrum_decoder, wiener_reconstruct


def _phantoms(n=16, size=32, seed=0):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    out = np.zeros((n, size, size), np.float32)
    for i in range(n):
        for _ in range(3):
            cy, cx = rng.uniform(0.2, 0.8, 2) * size
            w = rng.uniform(0.08, 0.2) * size
            out[i] += rng.uniform(0.5, 1) * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * w**2))
        out[i] /= out[i].max()
    return out


def test_power_spectrum_shape_and_positive():
    x = _phantoms()
    P = estimate_power_spectrum(x)
    assert P.shape == x.shape[1:]
    assert bool(jnp.all(P > 0))
    # Parseval: total power equals mean image energy (up to the eps floor)
    assert abs(float(P.sum()) - float(np.mean(np.sum(x**2, axis=(1, 2))))) < 1e-2 * float(P.sum())


def test_decoder_is_real_with_the_training_power():
    x = _phantoms()
    P = estimate_power_spectrum(x)
    decode = make_spectrum_decoder(P)
    w = jax.random.normal(jax.random.PRNGKey(0), (256, 2) + x.shape[1:])
    samples = jax.vmap(decode)(w)
    assert samples.shape == (256,) + x.shape[1:]
    assert bool(jnp.all(jnp.isfinite(samples)))
    energy = float(jnp.mean(jnp.sum(samples**2, axis=(1, 2))))
    assert abs(energy - float(P.sum())) < 0.25 * float(P.sum())


def test_wiener_full_sampling_recovers_the_image():
    x = _phantoms()
    P = estimate_power_spectrum(x)
    mask = jnp.ones(x.shape[1:])
    y = mask * fft2c(jnp.asarray(x[0]))
    out = wiener_reconstruct(y, mask, P, sigma=1e-4)
    assert bool(jnp.allclose(out["mean"], x[0], atol=1e-2))
    assert out["std"].shape == x.shape[1:]


def test_wiener_shrinks_noise_and_reports_constant_std():
    x = _phantoms()
    P = estimate_power_spectrum(x)
    mask = jnp.zeros(x.shape[1:]).at[:, ::2].set(1.0).at[:, 14:18].set(1.0)
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    sigma = 0.02
    noise = sigma * (jax.random.normal(k1, x.shape[1:]) + 1j * jax.random.normal(k2, x.shape[1:]))
    y = mask * (fft2c(jnp.asarray(x[0])) + noise)
    out = wiener_reconstruct(y, mask, P, sigma, n_samples=64, key=jax.random.PRNGKey(2))

    def err(est):
        return float(jnp.linalg.norm(jnp.asarray(est) - x[0]))

    assert err(out["mean"]) <= err(zero_filled(y, mask)) + 1e-3
    assert out["samples"].shape == (64,) + x.shape[1:]
    analytic = wiener_reconstruct(y, mask, P, sigma)["std"]
    assert bool(jnp.allclose(analytic, analytic.mean(), atol=1e-6))  # spatially constant
    assert abs(float(out["std"].mean()) - float(analytic.mean())) < 0.25 * float(analytic.mean())


def test_wiener_matches_hand_derived_posterior():
    # Hand-derived per-frequency posterior with sigma per real/imag component:
    # gain = P / (P + 2 sigma^2). Hermitian-symmetric y and mask keep everything real,
    # so fft2c(mean) recovers mean_k exactly and the gain is testable per frequency.
    n, sigma = 8, 0.5
    P = jnp.full((n, n), 3 * 2 * sigma**2)      # P = 3x complex noise var -> gain 0.75
    rng = np.random.default_rng(5)
    real_img = jnp.asarray(rng.standard_normal((n, n)), dtype=jnp.float32)
    y = fft2c(real_img)                          # Hermitian by construction
    mask = np.ones((n, n), np.float32)
    c = n // 2
    for a, b in [(1, 3), (6, 2)]:                    # knock out symmetric pairs (Hermitian mask)
        mask[a, b] = 0.0
        mask[(2 * c - a) % n, (2 * c - b) % n] = 0.0
    mask = jnp.asarray(mask)

    out = wiener_reconstruct(mask * y, mask, P, sigma)
    mean_k = fft2c(out["mean"])
    gain = 0.75
    assert bool(jnp.allclose(mean_k, mask * gain * y, atol=1e-4)), "gain must be P/(P + 2 sigma^2)"

    # analytic std: sqrt(sum(var_k) / (2 N)) with var_k = 2 P s2 /(P + 2 s2) observed, P unobserved
    s2c = 2 * sigma**2
    Pn, mn = np.asarray(P), np.asarray(mask)
    var_k = np.where(mn > 0, Pn * s2c / (Pn + s2c), Pn)
    expected_std = np.sqrt(var_k.sum() / (2 * var_k.size))
    assert abs(float(out["std"].mean()) - expected_std) < 1e-4


def test_wiener_single_sample_keeps_analytic_std():
    x = _phantoms()
    P = estimate_power_spectrum(x)
    mask = jnp.ones(x.shape[1:])
    y = mask * fft2c(jnp.asarray(x[0]))
    out = wiener_reconstruct(y, mask, P, 0.05, n_samples=1, key=jax.random.PRNGKey(0))
    assert float(out["std"].min()) > 0, "one sample must not produce a zero uncertainty map"


def test_decoder_reproduces_the_spectrum_per_bin():
    # P estimated from real images is Hermitian-symmetric, so the real-part decoder's
    # samples must reproduce it bin by bin (including the self-conjugate DC/Nyquist bins).
    x = _phantoms(n=16, size=16)
    P = estimate_power_spectrum(x)
    decode = make_spectrum_decoder(P)
    w = jax.random.normal(jax.random.PRNGKey(2), (512, 2, 16, 16))
    emp = jnp.mean(jnp.abs(fft2c(jax.vmap(decode)(w))) ** 2, axis=0)
    assert bool(jnp.all(jnp.abs(emp - P) < 0.35 * P + 1e-4)), "per-bin sample power must match P"


def test_wiener_all_zero_mask_returns_prior():
    x = _phantoms()
    P = estimate_power_spectrum(x)
    mask = jnp.zeros(x.shape[1:])
    out = wiener_reconstruct(mask * 0j, mask, P, 0.05)
    assert bool(jnp.allclose(out["mean"], 0.0, atol=1e-6))     # nothing measured -> prior mean
    assert float(out["std"].min()) > 0                         # ... with full prior uncertainty


def test_wiener_rejects_negative_sigma():
    import pytest

    x = _phantoms()
    P = estimate_power_spectrum(x)
    with pytest.raises(ValueError):
        wiener_reconstruct(jnp.zeros(x.shape[1:], jnp.complex64), jnp.ones(x.shape[1:]), P, -0.1)
