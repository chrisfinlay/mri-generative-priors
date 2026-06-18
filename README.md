# mri-generative-priors

Accelerated MRI reconstruction with a **generative prior** and **calibrated
uncertainty**, built on **JAX + Equinox + Optax + NumPyro** and managed with
**pixi**.

A learned prior (a β-VAE over knee slices) is plugged into a probabilistic model
in NumPyro. Given undersampled k-space, we recover the image two ways:

- **MAP** (point estimate) via SVI, and
- **the full posterior** via NUTS, which gives a per-pixel **uncertainty map**.

The reconstruction *is* a probabilistic model — `z ~ N(0, I)`, `x = decoder(z)`,
Gaussian likelihood on the observed k-space — which mirrors the radio-
interferometry workflow exactly, with a centred FFT in place of the NUFFT.

> ⚠️ **Data first.** fastMRI is patient-derived. Before anything else, read
> [`data/REGISTER_FIRST.md`](data/REGISTER_FIRST.md), register individually, and
> accept the Data Sharing Agreement. No patient data is committed to this repo.

## Quickstart

```bash
pixi install
pixi run check                      # prints JAX devices + 1-step VAE forward
# follow data/REGISTER_FIRST.md, then:
export FASTMRI_VAL_URL="<your personal singlecoil_val link>"
pixi run download                  # data/raw/*.h5
pixi run preprocess                # data/processed/*.npz  (128x128 magnitude)
pixi run lab                       # open the notebooks
pixi run test                      # shape / round-trip tests
```

If GPU training is slow, skip it: a pre-trained `checkpoints/vae_128.eqx`
ships with the repo (see [`CHECKPOINTS.md`](CHECKPOINTS.md)).

## What you implement (the `TODO`s)

Plumbing is given; **you implement the lines that teach the idea.** A
`solutions` branch carries a reference for every TODO — mentors reveal per
milestone if a team is stuck.

| File | Owner | What |
|------|-------|------|
| `src/mrigen/masks.py` | Team B | Cartesian undersampling masks (+ ACS band) |
| `src/mrigen/recon/operators.py` | Team B | forward `A(x)=M⊙F(x)`, adjoint, data consistency |
| `src/mrigen/models/vae.py` | all | the reparameterisation trick |
| `src/mrigen/recon/vae_numpyro.py` | Team B | the NumPyro `recon_model` body |
| `src/mrigen/metrics.py` | Team C | PSNR, NMSE |
| `src/mrigen/recon/classical.py` | optional | TV/L1 FISTA baseline |

Everything else — `fourier.py`, the VAE architecture + training loop, `viz.py`,
SSIM/diversity/calibration, the NumPyro MAP/NUTS drivers — is provided.

## Milestone map

| Notebook | Milestone |
|----------|-----------|
| `00_data_and_kspace.ipynb` | data, k-space, masks, the forward operator |
| `01_train_vae.ipynb` | train (or load) the β-VAE prior |
| `02_evaluate_prior.ipynb` | sample the prior, check reconstructions |
| `03_recon_map.ipynb` | **MAP reconstruction (Wednesday deliverable)** |
| `04_recon_posterior.ipynb` | NUTS posterior + uncertainty maps |
| `05_diffusion_stretch.ipynb` | *(stretch)* diffusion prior + DPS |
| `06_assemble_results.ipynb` | metrics tables + presentation figures |

See [`project4_tutorial.md`](../project4_tutorial.md) for the full walkthrough
and [`project4_schedule_plan.md`](../project4_schedule_plan.md) for the timeline.

## Layout

```
src/mrigen/        # library: fourier, masks, data, models, recon, metrics, viz
data/              # REGISTER_FIRST + downloader + preprocessing (no data committed)
notebooks/         # 00–06, the guided path
checkpoints/       # pre-trained weights (not data)
tests/             # shape / round-trip tests
```

## Stretch goals

Complex (2-channel) images using the real `kspace` measurement; a diffusion
prior with posterior sampling; variable-density / learned masks; calibration as
a quantitative deliverable; and the radio cross-over — swap `fft2c` for a NUFFT
and run the *same* pipeline on a non-Cartesian trajectory.
