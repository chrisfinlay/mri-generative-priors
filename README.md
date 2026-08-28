# mri-generative-priors

[![CI](https://github.com/chrisfinlay/mri-generative-priors/actions/workflows/ci.yml/badge.svg)](https://github.com/chrisfinlay/mri-generative-priors/actions/workflows/ci.yml)
[![Milestones](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fchrisfinlay%2Fmri-generative-priors%2Fbadges%2Fsolutions.json)](#what-you-implement-the-todos)

Accelerated MRI reconstruction with a **generative prior** and **calibrated
uncertainty**, built on **JAX + Equinox + Optax + NumPyro** and managed with
**pixi**.

> The **milestones** badge tracks how many of the `TODO`s below your team has
> implemented (`pixi run milestones` runs the same count locally). It turns
> green when every TODO's test passes. Forking? Update the two badge URLs above
> to point at your own `<owner>/<repo>`.

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

> 🪜 **New to JAX or Bayesian inference?** Start with [`PREP.md`](PREP.md): five short
> notebooks in `notebooks/prep/`, one a week before the school, that end with the whole
> project done in one dimension. The repo's `TODO`s are folded in as milestones.

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

| File | Thread | What |
|------|-------|------|
| `src/mrigen/masks.py` | reconstruction | Cartesian undersampling masks (+ ACS band) |
| `src/mrigen/recon/operators.py` | reconstruction | forward `A(x)=M⊙F(x)`, adjoint, data consistency |
| `src/mrigen/models/vae.py` | everyone | the reparameterisation trick |
| `src/mrigen/recon/vae_numpyro.py` | reconstruction | the NumPyro `recon_model` body |
| `src/mrigen/metrics.py` | evaluation | PSNR, NMSE |
| `src/mrigen/recon/classical.py` | optional (evaluation) | TV/L1 FISTA baseline |

Everything else — `fourier.py`, the VAE architecture + training loop, `viz.py`,
SSIM/diversity/calibration, the NumPyro MAP/NUTS drivers, the held-out split, the
power-spectrum prior and the model-agnostic evaluation (`evaluate.py`) — is
provided.

## Milestone map

| Notebook | Milestone |
|----------|-----------|
| `00_data_and_kspace.ipynb` | data, k-space, masks, the forward operator |
| `01_train_vae.ipynb` | train (or load) the β-VAE prior |
| `02_evaluate_prior.ipynb` | sample the prior, check reconstructions |
| `03_recon_map.ipynb` | **MAP reconstruction (Wednesday deliverable)** |
| `04_recon_posterior.ipynb` | NUTS posterior + uncertainty maps |
| `05_diffusion_stretch.ipynb` | *(stretch)* diffusion prior + DPS |
| `06_evaluate_models.ipynb` | **evaluate every model on one protocol** (held-out split, table, curve, calibration, worst case) |

See [TUTORIAL.md](TUTORIAL.md) for the full walkthrough.

## How we'll work at the school

**Three threads, not three teams.** The work has three parts — the *prior*
(notebooks 01–02), the *reconstruction* (03–04) and the *evaluation* (06) — and
the Friday talk follows them. With five people, everyone does the whole path:

- **Mon–Wed, one path for all.** Notebook 00 → load the checkpoint → a MAP
  reconstruction of *your own* slice at *your own* R that beats zero-filled
  (the Wednesday deliverable, five times over). Work in pairs; pairs rotate at
  every 17:30 stand-up.
- **Thursday, split by depth.** Each pair or person goes deep on one thread
  and reports on it Friday: the NUTS posterior and uncertainty maps; the
  evaluation and calibration; a second model (TV/L1, the power-spectrum prior,
  the diffusion or complex-image stretch) through the same evaluation.
- **Evaluation is shared.** Every model goes through `06_evaluate_models.ipynb`;
  each person owns one row of the final table.

**A lead for each day.** One student leads each day, Mon–Fri, so everyone leads
once (the rota is volunteered on Monday morning). The lead's job is about half an
hour of the day: a five-minute kickoff (today's milestone, who pairs with whom);
keeping the critical path visible and calling the mentor when a pair has been
stuck for 45 minutes; running the 17:30 stand-up (done / blocked / next, every
pair speaks, lead summarises); and five lines in `LOG.md` — date, lead, what
shipped, what broke, the decision for tomorrow. Friday's lead runs the rehearsal
and keeps time in the talk. The mentor coaches the lead rather than running the
room.

## Layout

```
src/mrigen/        # library: fourier, masks, data, models, recon, metrics, evaluate, viz
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
