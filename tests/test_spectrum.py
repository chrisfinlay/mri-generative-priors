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
    assert abs(float(out["std"].mean()) - float(analytic.mean())) < 0.5 * float(analytic.mean())
