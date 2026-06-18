# Tutorial — Data-Driven Priors for MRI Reconstruction

The guided path through this repo. Read a section, then run the matching
notebook in `notebooks/`. Code blocks are sketches — the `# TODO` lines are the
ones you implement (see the ownership table in `README.md`). The conventions all
the code obeys (FFT, shapes, normalisation, the forward model) are pinned in
`CLAUDE.md`; this tutorial explains the *ideas* behind them.

```
pixi install
pixi run check                 # confirm JAX sees your GPU
# follow data/REGISTER_FIRST.md, then download + preprocess
pixi run lab                   # open the notebooks
```

---

## 0. The big picture in one paragraph

An MRI scanner doesn't photograph you. It measures **k-space** — the 2D Fourier
transform of the image — and measuring all of it is slow. Measure *less* and the
scan is faster, but the naive image is wrecked by aliasing. Recovering a clean
image from incomplete k-space is an **ill-posed inverse problem**: many images
fit the data, so we need a *prior* telling us which images are plausible. We
**learn** that prior with a generative network and use it inside a **Bayesian**
reconstruction, which returns not just an image but a per-pixel measure of how
*uncertain* it is.

If you know radio interferometry, this is your problem with the NUFFT replaced by
a plain FFT. If you don't: you'll build the whole thing from FFTs and a small net.

---

## Part 1 — k-space, the image, and aliasing · `00_data_and_kspace.ipynb`

### 1.1 What the scanner measures

The image `x` and the measurement `k` are a Fourier pair (helpers in
`src/mrigen/fourier.py`):

```
k = fft2c(x)        # k-space
x = ifft2c(k)       # image
```

- The **centre** of k-space holds low spatial frequencies — overall contrast.
- The **edges** hold high frequencies — fine detail and edges.

> **Radio analogy.** k-space is the (u, v) plane. Centre = short baselines;
> edges = long baselines. Fully-sampled k-space ↔ perfect uv-coverage.

### 1.2 Accelerating = undersampling

Keep only some k-space, described by a binary **mask** `M` (you implement this in
`masks.py`). The fraction kept is `1/R`, the *acceleration* (4×, 8×, 16×). Always
keep a fully-sampled **ACS** band at the centre — the lowest frequencies matter
most. The measurement is:

```
y = M ⊙ fft2c(x) + noise
```

### 1.3 The naive reconstruction, and why it fails

```python
x_zf = ifft2c(y).real          # "zero-filled" reconstruction
```

Inverse-transforming the masked k-space (zeros where you didn't measure) gives
the **zero-filled** image — full of aliasing. This is your baseline to beat.

> **Radio analogy.** This is the *dirty image*: the true sky convolved with the
> dirty beam. Same maths, same artefacts.

**Tasks:** plot `|k|` (log) and `|x|`; build a mask at R = 4 (`masks.py` TODO,
Team B); produce and plot the zero-filled recon. *See the aliasing.*

---

## Part 2 — Two things about the data you must understand

### 2.1 The provided files

Each single-coil knee volume (HDF5) gives:

- `kspace` — emulated single-coil k-space at the **native** size.
- `reconstruction_esc` — inverse FFT of that k-space, **centre-cropped to
  320×320** (the standard target). Because of the crop, `fft2c(reconstruction_esc)
  ≠ kspace` — the crop discards information, so the two are *not* a clean FFT pair.
- `reconstruction_rss` — a target from the original multi-coil data.

Because we do **retrospective undersampling**, we mostly *synthesise* k-space
ourselves: take a clean image, crop/resize to 128², FFT it, apply the mask. The
provided `kspace` is your reference for "what a real measurement looks like".

### 2.2 Why the images are complex (and what phase means)

The MR signal is intrinsically complex (quadrature detection), so
`reconstruction_esc` is complex-valued. Its **magnitude** is the anatomy you want;
its **phase** comes from field inhomogeneity, fat/water off-resonance, receive-coil
phase, flow, and tissue susceptibility — real physics, but mostly *nuisance* for
anatomical reconstruction. So we make a standard simplification: **train on
magnitude images and treat them as real-valued.**

> **Radio analogy, sharpened.** A real image with complex Fourier samples is
> *exactly* the radio situation — real sky, complex visibilities. The magnitude
> path is the radio-identical version of MRI. Modelling the phase too (the
> *complex* path) is the one wrinkle radio doesn't have — that's the stretch goal.

**Tasks:** take `|reconstruction_esc|`, centre-crop and resize to 128², normalise
to [0, 1]. (Normalisation lives in `data.py` and must be applied *identically* at
train and recon time — see `CLAUDE.md` Contract 2.) These are your images `x`.

---

## Part 3 — Learning the prior with a VAE · `01_train_vae.ipynb`, `02_evaluate_prior.ipynb`

### 3.1 The idea

A generative model maps a simple latent to a realistic image:

```
z ~ N(0, I)        # latent code, dim 128–256 (keeps NUTS tractable later)
x = decoder(z)     # a realistic knee image
```

The decoder *is* the prior: the images it can produce, weighted by the Gaussian
on `z`, are `p(x)`. Anything off that manifold is implausible.

### 3.2 The VAE in one screen

An encoder maps an image to a latent Gaussian `q(z|x) = N(μ, σ²)`; sample `z` via
the **reparameterisation trick** (your TODO in `models/vae.py`); the decoder
reconstructs. Train by maximising the ELBO:

```python
def reparam(mu, logvar, key):
    # TODO: mu + exp(0.5 * logvar) * standard_normal(key)
    ...

# loss = mean((x - decoder(z))**2)  +  beta * KL(q(z|x) ‖ N(0, I))
```

### 3.3 What to expect (a teaching point)

VAE samples will be **smooth/blurry** — the well-known quality cost of VAEs, and
exactly the *quality vs uncertainty* trade-off this project is about. Knobs:
latent dimension, β, training length. **If training is slow, load
`checkpoints/vae_128.eqx`** and move on — the prior is not the critical path.

### 3.4 Evaluating the prior (`02_evaluate_prior.ipynb`)

- **Quality:** do samples look like knees? (Eyeball + the slide checklist.)
- **Diversity:** pairwise SSIM across samples (low = diverse); latent
  interpolations should morph smoothly.
- **Speed:** time per sample — matters when you compare to diffusion later.

**Tasks:** finish `reparam`; train or load the VAE; sample grid; diversity score;
a latent interpolation between two codes.

---

## Part 4 — Bayesian reconstruction · `03_recon_map.ipynb`, `04_recon_posterior.ipynb`

The heart of the project, and the worked example from the probabilistic-
programming lecture.

### 4.1 The posterior

```
p(x | y) ∝ p(y | x) · p(x)
```

- **Likelihood** `p(y | x)`: Gaussian — the measured k-space should match
  `M ⊙ fft2c(x)` up to noise σ.
- **Prior** `p(x)`: the learned model, via `x = decoder(z)`, `z ~ N(0, I)`.

We **infer the latent `z`, not the image.** This bakes the prior in (every `z`
decodes to a plausible image) and shrinks a million-pixel problem to ~128 dims.

### 4.2 The model (NumPyro) — your TODO in `recon/vae_numpyro.py`

```python
def recon_model(y_obs, mask, decoder, latent_dim, sigma):
    z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), 1.0))
    x = decoder(z)                       # prior pushforward (real image)
    k = mask * fft2c(x)                  # forward operator A(x)
    obs = mask.astype(bool)
    # TODO: observe real & imaginary parts of the measured k-space
    numpyro.sample("y_re", dist.Normal(k.real[obs], sigma), obs=y_obs.real[obs])
    numpyro.sample("y_im", dist.Normal(k.imag[obs], sigma), obs=y_obs.imag[obs])
```

> `decoder` must be a **pure** function for inference — it's built with
> `eqx.partition`/`eqx.combine` so the trained weights are baked in (see
> `CLAUDE.md` Contract 4).

### 4.3 MAP — the point estimate (`03_recon_map.ipynb`)

Find the single most probable `z`, then decode it:

```python
guide = numpyro.infer.autoguide.AutoDelta(recon_model)
svi = numpyro.infer.SVI(recon_model, guide, optax.adam(1e-2), loss=Trace_ELBO())
# run; decode the learned z to x_map
```

Compare `x_map` to zero-filled and TV baselines at R = 4 and 8. **Wednesday
deliverable.**

### 4.4 Posterior — uncertainty for free (`04_recon_posterior.ipynb`)

Run **NUTS** over `z`, decode every sample, take the per-pixel **mean** (the
reconstruction) and **std** (the uncertainty):

```python
mcmc = numpyro.infer.MCMC(numpyro.infer.NUTS(recon_model),
                          num_warmup=500, num_samples=500)
mcmc.run(key, y_obs, mask, decoder, latent_dim, sigma)
zs = mcmc.get_samples()["z"]
imgs = jax.vmap(decoder)(zs)
x_mean, x_std = imgs.mean(0), imgs.std(0)
```

The `x_std` map is the headline result: *where is the reconstruction
trustworthy?* Expect higher uncertainty at fine structures and high acceleration.

> If NUTS is slow or mixes badly (check r-hat ≈ 1), fall back to MAP + a Laplace
> approximation for a cheap uncertainty proxy — and say so in the talk.

**Tasks:** complete the likelihood; run MAP; run NUTS; produce reconstruction,
error, and uncertainty maps; sweep R ∈ {4, 8, 16}.

---

## Part 5 — Evaluation · `06_assemble_results.ipynb` (Team C, ongoing)

Two questions, two kinds of metric:

- **Reconstruction quality (quantitative):** PSNR, SSIM, NMSE vs the
  fully-sampled ground truth, as a function of acceleration, across three methods
  — zero-filled, classical TV/L1, the deep prior. One table + one curve.
- **Reconstruction quality (qualitative/medical):** does the recon preserve the
  structures a clinician cares about? Look at **error maps**, not just numbers —
  a high PSNR can still smear a small but important feature.
- **Uncertainty calibration:** bin pixels by predicted std, plot mean error per
  bin. Good UQ ⇒ error rises with uncertainty.
- **Prior intrinsics:** quality / diversity / speed from Part 3.

(`PSNR`/`NMSE` are your TODOs in `metrics.py`, Team C; SSIM, diversity, and
calibration helpers are provided.)

---

## Part 6 — Stretch: diffusion prior · `05_diffusion_stretch.ipynb`

A score-based model learns `∇ₓ log p(x)` at many noise levels and generates by
reversing a noising process — typically **sharper and more diverse** than a VAE,
but slower. Reconstruction alternates reverse-diffusion steps with data
consistency on the measured k-space (posterior sampling). Compare to the VAE on
the same quality / diversity / speed axes — that comparison is a strong result on
its own.

---

## What "done" looks like

Minimal complete project: a trained-or-loaded VAE, a working MAP reconstruction
beating zero-filled at R = 4, and one figure (recon + error + uncertainty).
Everything past that — posterior sampling, the R-sweep, calibration, diffusion,
the complex path — is upside. **Aim for the minimal complete result by Wednesday
evening; spend Thursday making it deep.**
