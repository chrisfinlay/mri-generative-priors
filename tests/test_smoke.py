"""End-to-end vertical-slice smoke test on SYNTHETIC data (no fastMRI needed).

Exercises the whole magnitude-path pipeline at once: mask -> forward operator
-> measurement -> NumPyro recon_model -> MAP via SVI -> decode -> metrics. The
synthetic ground truth is drawn from the prior's own range (``x = decode(z*)``)
so it is exactly representable; MAP must then clearly beat the zero-filled
baseline. Uses ``skip_if_unimplemented`` so it skips (not fails) on ``main``
where the TODOs still raise, and runs for real on ``solutions``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mrigen.masks import equispaced_mask
from mrigen.metrics import psnr
from mrigen.models.vae import VAE, make_decoder_fn
from mrigen.recon.classical import zero_filled
from mrigen.recon.operators import forward
from mrigen.recon.vae_numpyro import reconstruct_map
from conftest import skip_if_unimplemented

LATENT = 128


@skip_if_unimplemented
def test_map_beats_zero_filled_on_synthetic():
    vae = VAE(latent_dim=LATENT, key=jax.random.PRNGKey(0))
    decode = make_decoder_fn(vae)

    # Synthetic image that lies in the prior's range, so MAP can recover it.
    z_true = jax.random.normal(jax.random.PRNGKey(1), (LATENT,))
    x = decode(z_true)

    mask = equispaced_mask((128, 128), acceleration=4)
    sigma = 0.02
    nk = jax.random.PRNGKey(2)
    noise = sigma * (
        jax.random.normal(nk, (128, 128)) + 1j * jax.random.normal(jax.random.PRNGKey(3), (128, 128))
    )
    y = forward(x, mask) + mask * noise

    x_zf = zero_filled(y, mask)
    x_map, _ = reconstruct_map(y, mask, vae.decoder, LATENT, sigma=sigma, steps=200, lr=2e-2)

    dr = float(x.max() - x.min())
    p_zf = psnr(np.asarray(x), np.asarray(x_zf), data_range=dr)
    p_map = psnr(np.asarray(x), np.asarray(x_map), data_range=dr)

    assert p_map > p_zf + 5.0          # MAP clearly beats the baseline
    assert p_map > 20.0                # and is a genuinely good reconstruction
    assert jnp.all(jnp.isfinite(x_map))
