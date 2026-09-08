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

**Tasks:** plot `|k|` (log) and `|x|`; build a mask at R = 4 (`masks.py` TODO); produce and plot the zero-filled recon. *See the aliasing.*

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

- **Quality:** do samples look like knees? (Eyeball, then the visual checklist in Part 5.)
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

### 4.2 The model (NumPyro) — the `recon_model` TODO in `recon/vae_numpyro.py`

```python
def recon_model(y_obs, mask, decoder, latent_dim, sigma):
    z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), 1.0))
    x = decoder(z)                       # prior pushforward (real image)
    k = mask * fft2c(x)                  # forward operator A(x)
    obs = mask.astype(bool)
    # TODO: observe real & imaginary parts of the measured k-space.
    # The mask is a *traced* array under NUTS/SVI, so boolean indexing
    # (k.real[obs]) raises NonConcreteBooleanIndexError — restrict the
    # likelihood with .mask(obs) over the full array instead.
    numpyro.sample("y_re", dist.Normal(k.real, sigma).mask(obs), obs=y_obs.real)
    numpyro.sample("y_im", dist.Normal(k.imag, sigma).mask(obs), obs=y_obs.imag)
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

## Part 5 — Evaluation: one protocol for every model · `06_evaluate_models.ipynb`

Evaluation is not a final step; it is how you know whether anything you built
works, and it is what the Friday talk is made of. The rule that makes it worth
doing: **every method is judged on exactly the same problem.** `mrigen.evaluate`
enforces that, and it is model-agnostic — the methods in this repo and any model
you add go through it identically.

### 5.1 The interface: a reconstructor

```python
recon(y_obs, mask, sigma) -> ev.Recon(mean, std=None, samples=None)
```

That is all a method has to be. Adapters for the repo's methods ship in
`evaluate.py` (`zero_filled_recon`, `tv_recon`, `wiener_recon`, `map_recon`,
`posterior_recon`); a new model needs a three-line adapter of the same shape.
`map_recon` / `posterior_recon` take *any* decoder — a VAE, a diffusion model
with a deterministic sampler, the power-spectrum decoder — so a prior with a
decoder needs no adapter at all.

### 5.2 The protocol (say it in the talk)

1. **Held-out slices only** — `FastMRISlices(root, split="test")`. The volumes in
   `mrigen.data.HELDOUT_VOLUMES` are excluded from the checkpoint's training and
   must be excluded from yours. Numbers on training slices are not results.
2. **Same mask, same noise draw, same σ for every method**, seeded per (slice, R).
3. **PSNR, SSIM, NMSE** against the fully-sampled truth, `data_range = 1`.
4. **Effective acceleration** `R_eff = M.size / M.sum()` in every table header —
   the ACS band makes a nominal 4× about 3.3×.
5. **Seconds per reconstruction**, measured after a warm-up call (JIT).
6. **Calibration** for every method with a `std`: pool the pixels over slices, bin
   by predicted std, plot actual |error| per bin against the diagonal.
7. **Mean ± std over slices.** A difference smaller than the ± is "comparable".
8. **The worst case**, per method, as a panel — show one in the talk.

### 5.3 Two kinds of quality

- **Quantitative (statistical):** the table and the PSNR-vs-R curve, across
  zero-filled, TV/L1, the power-spectrum prior, and the deep prior (MAP and
  posterior mean). Where does each win, and by more than the spread?
- **Qualitative (clinical):** look at error maps, not just numbers — a high PSNR
  can still smear a small structure that matters. The visual checklist:
  anatomy preserved (bone edges, cartilage surfaces, menisci, ligaments)?
  residual aliasing, ringing, blur? **hallucinated** structure that is not in
  the truth? does the uncertainty map light up where the error is?

### 5.4 Uncertainty: is it honest?

A per-pixel std is only useful if it is *calibrated*. Expect two things: the
power-spectrum prior's std is flat (a stationary prior cannot say *where* it is
unsure), and the VAE posterior is **overconfident** — the decoder cannot
represent a held-out slice exactly, and the model has no term for "my prior
cannot make this image", so it reports tight error bars around the nearest
image it *can* make. That is model misspecification; name it, don't hide it.
(You met it in one dimension in prep notebook 5.)

### 5.5 Adding a model — and comparing it fairly

Route A, a prior with a decoder: write a pure `decode(z)`, pick the latent
shape, hand both to `map_recon` / `posterior_recon`. Route B, a method that is
its own algorithm (DPS, an unrolled network, a classical solver): write the
adapter. Then:

1. fit it on `split="train"` only, and state what it saw;
2. run it on the **same** slices, R and σ as everyone else — one `methods` dict,
   one `evaluate` call;
3. report `R_eff`, seconds and the ± next to the others; include it in the
   calibration plot if it has a std, and say so if it does not;
4. show its worst case;
5. evaluate the **prior itself** (notebook 02): sample quality, diversity
   (mean pairwise 1 − SSIM), time per sample. A prior can win on reconstruction
   and lose on diversity; both are results.

The worked example is the **power-spectrum prior** (`recon/spectrum.py`): a
stationary Gaussian whose spectrum is learnt from the training slices in one
line, used both as a decoder for `recon_model` and as a closed-form Wiener
reconstructor. It teaches the lesson every new model should be measured
against: a prior that is diagonal in k-space cannot fill in k-space it never
measured, so it barely beats zero-filling — de-aliasing needs a prior that
couples frequencies, i.e. knows about spatial structure (sparsity, or a
learned decoder).

(`PSNR`/`NMSE` are your TODOs in `metrics.py`; SSIM, diversity and calibration
helpers are provided.)

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
