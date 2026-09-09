# Before the school — the prep ladder

Six rungs, one a week, about nine hours in total. Each rung ends with a **check** that tells you it is
done, so you never have to wonder. Everything runs on a laptop CPU; only the optional real-data step
needs the fastMRI download. At the school you will have a shared GPU server running JupyterLab.

The goal is not to finish the project early. It is to arrive in Cape Town having *already* written a
Bayesian reconstruction — in one dimension — so that Monday afternoon starts with knees, not syntax.

| Week of | Rung | Do | Done when |
|---|---|---|---|
| 24 Aug | **0 · Set up** | Register at [fastmri.med.nyu.edu](https://fastmri.med.nyu.edu/) *today* (approval takes time). Install [pixi](https://pixi.sh/), clone this repo, `pixi install`, `pixi run check`, `pixi run test`. | `pixi run test` says **20 passed, 11 skipped** — the skips are yours to turn green |
| 31 Aug | **1 · NumPy → JAX** | `notebooks/prep/01_numpy_to_jax.ipynb`, then implement `psnr` and `nmse` in `src/mrigen/metrics.py` | `pixi run milestones` says **2/8** |
| 7 Sep | **2 · Fourier & k-space** | `notebooks/prep/02_fourier_and_kspace.ipynb`, then `masks.py` and `recon/operators.py` | `pixi run milestones` says **8/8** |
| 14 Sep | **3 · Bayes by hand** | `notebooks/prep/03_bayes_by_hand.ipynb` — prior, likelihood, posterior, MAP, uncertainty, all on a grid | three checks print OK; you can say what MAP vs posterior mean is |
| 21 Sep | **4 · First NumPyro model** | `notebooks/prep/04_first_numpyro_model.ipynb` — the coin, two pixels, then a 1-D MRI with NUTS and MAP | the posterior beats zero-filling in 1-D and `r_hat ≈ 1` |
| 28 Sep | **5 · A learned prior** | `notebooks/prep/05_learned_prior_in_1d.ipynb`, then `reparameterise` in `models/vae.py` | a 1-D reconstruction with an uncertainty band from a prior you trained |
| ahead? | *6 · `recon_model`* | `src/mrigen/recon/vae_numpyro.py`, then `pixi run slice` | `outputs/slice.png`: MAP beats zero-filled on synthetic phantoms |

Rungs 1, 2 and 5 include the repo's own milestone `TODO`s, so the badge in the README tracks your
progress. Rung 6 is the school's Wednesday deliverable — you do not need it before October, but its 1-D
twin is exactly what you write in rungs 4 and 5.

## How to work through a notebook

```bash
pixi run lab            # opens JupyterLab; the prep notebooks are in notebooks/prep/
```

- Run every cell in order and read the comments — they are the explanation.
- Exercise cells contain `# YOUR CODE HERE` and `...`. Replace the `...`, run the cell, then run the
  **check** cell below it. No `AssertionError` = done.
- Under each exercise sits a **collapsed hint** (click to expand). Try honestly first — 15 minutes of
  being stuck teaches more than the hint does — and treat opening it as spending a life, not as step one.
- When a notebook asks you to implement a function *in the repo* (`metrics.py`, `masks.py`, ...), edit the
  file, then re-run the notebook's bridge cell: it reloads the module for you. If it still shows the old
  behaviour, restart the kernel.
- Stuck for more than an hour? Open an issue on GitHub with the cell you ran and the full error. Someone
  else will hit the same thing.

## Platforms

Linux and macOS work out of the box. On Windows, use [WSL 2](https://learn.microsoft.com/windows/wsl/)
and run everything inside it. No GPU is needed for any rung — the largest computation is a minute of
CPU in rung 5.

## What each rung prepares you for

| rung | school notebook |
|---|---|
| 1 · JAX | everything |
| 2 · k-space, masks, operators | `00_data_and_kspace` |
| 3–4 · Bayes, NumPyro, NUTS, MAP | `03_recon_map`, `04_recon_posterior` |
| 5 · VAE, reparameterisation, a learned decoder | `01_train_vae`, `02_evaluate_prior` |
| 6 · `recon_model` | the Wednesday deliverable |

## Background reading, if you want more

None of this is required; `TUTORIAL.md` explains every idea at the depth the project needs.

- **The physics (rung 2):** [What is k-space?](https://mriquestions.com/what-is-k-space.html) ·
  [fastMRI](https://arxiv.org/abs/1811.08839), the dataset paper.
- **The inference (rungs 3–4):** [NumPyro: Bayesian regression](https://num.pyro.ai/en/stable/tutorials/bayesian_regression.html) ·
  Betancourt, [A conceptual introduction to HMC](https://arxiv.org/abs/1701.02434).
- **The prior (rung 5):** Kingma & Welling, [An introduction to VAEs](https://arxiv.org/abs/1906.02691) ·
  Bora et al., [Compressed sensing using generative models](https://arxiv.org/abs/1703.03208) — "infer z, not x".
- **The tools (rung 1):** [JAX quickstart](https://docs.jax.dev/en/latest/quickstart.html) ·
  [the sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) ·
  [Equinox](https://docs.kidger.site/equinox/) · [pixi](https://pixi.sh/).
