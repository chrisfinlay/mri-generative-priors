# CLAUDE.md

Guidance for Claude Code sessions on **mri-generative-priors**. Read this first.
Keep changes consistent with the contracts below — most bugs in this project come
from breaking one of them silently.

---

## What this repo is

A teaching codebase for the Big Data Africa School (Project 4). Students recover
an MRI image from **undersampled k-space** by plugging a learned generative prior
(a β-VAE) into a probabilistic model in NumPyro, and read off **calibrated
uncertainty** from the posterior.

Stack: **JAX + Equinox + Optax + NumPyro**, managed with **pixi**. No PyTorch.

Audience: 4th-year-BSc-and-up, **numpy-native**, mixed ML experience. Code must be
readable and well-commented — clarity over cleverness, especially near the TODOs.

Design source of truth: `TUTORIAL.md` (the walkthrough), plus the schedule and
slide-outline docs. If a change contradicts those, flag it rather than diverging.

---

## The pedagogical structure (do not break this)

- **`main`** ships with the conceptual lines removed as `# TODO`s (see the table
  in `README.md`). Each TODO has a test; the `pixi run milestones` badge counts
  how many pass.
- **`solutions`** carries a working reference for every TODO.
- **Workflow: implement on `solutions` first, verify it runs end-to-end, then
  strip the conceptual lines back out onto `main`.** Build complete, then carve.
- **Never merge `solutions` into `main`.** Never delete a TODO's test.
- A TODO is a *single conceptual idea* a student can implement in a few lines
  (the reparam trick, the mask, the forward operator, the likelihood, PSNR).
  Plumbing around it stays given and working.

---

## Contract 1 — the FFT convention (the #1 bug source)

There is **one** Fourier convention, in `src/mrigen/fourier.py`, used *everywhere*
(training targets, forward operator, likelihood, baselines, viz). Centred and
orthonormal:

```python
def fft2c(x):
    return jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )

def ifft2c(k):
    return jnp.fft.fftshift(
        jnp.fft.ifft2(jnp.fft.ifftshift(k, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )
```

Required properties, each asserted in `tests/`:
- **Exact inverse:** `ifft2c(fft2c(x)) ≈ x`.
- **Unitary / Parseval:** `‖fft2c(x)‖₂ ≈ ‖x‖₂` (so the noise scale `sigma` means
  the same thing in both domains — this is why we use `norm="ortho"`).
- **Centred:** DC at the array centre, so masks and the ACS band are defined
  about the centre. Operates on the last two axes only.

Do not introduce a second FFT helper, a different `norm`, or a stray `fftshift`.

## Contract 2 — shapes, dtypes, normalisation

- Image `x`: **real `float32`, shape `(H, W) = (128, 128)`, values in `[0, 1]`.**
  Normalisation lives in `data.py` and **must be applied identically at training
  and reconstruction time.** Store/return the per-slice scale if you normalise by
  it — recon needs the same scale. Inconsistent normalisation = silent garbage.
- k-space: **complex64, `(H, W)`.**
- Mask `M`: `{0, 1}` `float32`, `(H, W)`, with a fully-sampled centre ACS band.
  Acceleration `R = M.size / M.sum()`.
- Batches stack on a leading axis; never assume a channel axis on the magnitude
  path (complex/2-channel is the stretch path, behind a flag).

## Contract 3 — the forward model (magnitude path is primary)

```
x = decoder(z)            # real, non-negative-ish image
k = fft2c(x)              # complex k-space
y = M * k + noise         # measurement; noise ~ complex Gaussian, std `sigma`
```

- Data consistency: keep observed entries, trust the estimate elsewhere —
  `ifft2c(M * y + (1 - M) * fft2c(x_est)).real`.
- The **likelihood observes the real and imaginary parts** of `y` at masked
  entries with `Normal(·, sigma)`. Same `fft2c` as above — no exceptions.
- Complex (2-channel) path uses the *provided* `kspace` array instead of
  synthesising it; everything else is unchanged. Keep it behind a flag; do not
  let it complicate the magnitude path.

## Contract 4 — Equinox purity for inference

NumPyro needs a **pure** `z -> x` function it can differentiate through.

- Split params from static with `eqx.partition(model, eqx.is_array)`; rebuild
  inside a closure with `eqx.combine`. Expose `make_decoder_fn(model) -> (z -> x)`
  with params baked in.
- Use `eqx.filter_jit` / `eqx.filter_grad`, not bare `jax.jit`, on anything
  holding a model.
- Keep `latent_dim` in **128–256** so NUTS mixes in reasonable time.

---

## Running things

```
pixi install
pixi run check         # JAX devices + 1-step VAE forward (run before anything)
pixi run download      # needs FASTMRI_VAL_URL; writes data/raw/*.h5
pixi run preprocess    # data/processed/*.npz (128x128 magnitude)
pixi run test          # shape / round-trip / Parseval tests
pixi run lint          # ruff
pixi run milestones    # count passing TODO tests (drives the badge)
pixi run lab           # Jupyter
```

`pixi run test` must pass on `solutions`. On `main`, the TODO tests are expected
to fail until students implement them — that is what the milestones badge counts.

---

## Data rules (patient-derived — strict)

- **Never commit data.** `data/raw/` and `data/processed/` are git-ignored; keep
  them so. No `.h5`, `.npz`, or images in any commit.
- Checkpoints (`checkpoints/*.eqx`) are model weights, not data — those are fine.
- The downloader reads a per-user link from `FASTMRI_VAL_URL`; it must not embed
  or log any link. fastMRI requires individual registration (`data/REGISTER_FIRST.md`).
- Expected fastMRI single-coil schema (validate the loader against it):
  `kspace` (complex, native size), `reconstruction_esc` (complex, 320×320),
  `reconstruction_rss` (320×320). We use `|reconstruction_esc|` → crop/resize to 128².

---

## Dependencies & style

- Add deps to `pixi.toml`, not ad-hoc `pip`. Keep the set minimal: JAX (CUDA),
  Equinox, Optax, NumPyro, h5py, numpy, matplotlib, scikit-image (SSIM only),
  jupyter. Justify anything new.
- Public functions get a docstring and type hints. Comment the *why* on the
  conceptual lines; keep the lines around a TODO simple enough that a student can
  read them to understand what their TODO must produce.
- Small, focused functions and modules. Don't over-engineer — no config
  frameworks, no abstract base classes for a one-off.

---

## Before you commit

1. `pixi run test` (on `solutions`, all green; on `main`, only TODO tests red).
2. `pixi run lint`.
3. No data files staged.
4. If you touched a contract above, update its test and say so in the commit.
5. Small, single-purpose commits with clear messages.

## Per-session scope (don't one-shot the repo)

Work in coherent chunks, in this order:
1. Foundations — `fourier`, `masks`, `data`, `metrics`, `viz`, tests; Notebook 00.
2. Vertical slice — tiny VAE → `recon_model` → MAP on synthetic, end-to-end.
3. VAE proper + training loop (the real checkpoint is trained by the mentor).
4. Reconstruction — operators, MAP (SVI), NUTS + uncertainty maps.
5. Evaluation — classical baselines, metrics tables, calibration; Notebook 06.
6. (Stretch) diffusion prior + posterior sampling.

Prove each chunk runs before widening the next. When in doubt about a contract or
a design choice, ask rather than diverge from `TUTORIAL.md`.
