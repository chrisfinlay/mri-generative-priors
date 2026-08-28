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
| Optimiser | Adam, lr 1e-3 |
| Held-out recon PSNR / SSIM | _fill in after pre-training_ |

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
