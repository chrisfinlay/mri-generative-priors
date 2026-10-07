# Checkpoints

Pre-trained model weights ship here so a team can start reconstructing even if
GPU training is slow. **These are model weights, not patient data**, and are
safe to redistribute under the fastMRI Data Sharing Agreement.

## `vae_128.eqx`

| Field | Value |
|-------|-------|
| Model | convolutional β-VAE (`mrigen.models.vae.VAE`) |
| Resolution | 128 × 128 magnitude |
| Latent dim | 128 |
| β | 1.0 |
| Training data | knee magnitude slices from `knee_singlecoil_val`, **train split only** — the volumes in `mrigen.data.HELDOUT_VOLUMES` (`file1000593`, `file1002067`, the first two in the archive) were excluded |
| Training data (server) | on the school server, `singlecoil_train` **train split only** (see `SERVER.md`); `singlecoil_val` is the untouched test set |
| Optimiser | Adam, lr 1e-3, gradient norm clipped at 1.0; batch 32, 50 epochs |
| Held-out recon PSNR / SSIM | server checkpoint, autoencoding (`decoder(encoder mean)`) on the server `test` split (all 7,135 `singlecoil_val` slices; SSIM on every 10th): **23.74 ± 2.39 dB / 0.578 ± 0.140** |

### β sweep (school server only)

Trained identically on the server `train` split. All four are in
`/shared/checkpoints/`, and `pixi run server-setup` copies them into your clone.
Load one with `load_model("checkpoints/vae_128_beta0.1.eqx", latent_dim=128)`.

| file | β | val PSNR (dB) | test PSNR (dB) | test SSIM |
|------|---|---------------|----------------|-----------|
| `vae_128.eqx` = `vae_128_beta1.0.eqx` | 1.0 | 23.73 | 23.74 | 0.578 |
| `vae_128_beta0.3.eqx` | 0.3 | 25.17 | 25.14 | 0.622 |
| `vae_128_beta0.1.eqx` | 0.1 | 26.18 | 26.10 | 0.651 |
| `vae_128_beta0.03.eqx` | 0.03 | 26.48 | 26.40 | 0.660 |

Smaller β reconstructs better but regularises the latent space less, and a
reconstruction *prior* needs that regularisation: an autoencoding score alone
should not pick β. Choose on `val` by reconstruction quality *from undersampled
k-space* (notebooks 03/04) and report `test` once.

Load it:

```python
from mrigen.train_vae import load_model
vae = load_model("checkpoints/vae_128.eqx", latent_dim=128)
```

Compare your own training run against these numbers — if you can beat them by
tuning β / latent dim / epochs, even better. Train with
`python -m mrigen.train_vae` (it uses `split="train"` by default) and **evaluate
only on `FastMRISlices(split="test")`** — see `06_evaluate_models.ipynb`.

## `score_128.eqx` *(stretch)*

Small UNet score model, same data and resolution. Only present if the diffusion
stretch goal was pre-trained.

> **Note:** checkpoints are produced by the mentor before the school and dropped
> into this directory. They are git-ignored by default (see `.gitignore`) so the
> repo stays small; the mentor distributes them out of band or via a release.
