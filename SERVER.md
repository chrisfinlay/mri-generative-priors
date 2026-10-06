# Running on the school GPU server

**Optional.** This page applies only to the shared school server, where the
fastMRI data is already downloaded into `/shared`. On your own machine, follow
the Quickstart in [`README.md`](README.md); none of this is needed.

## Students: once, in your own clone

```bash
git clone https://github.com/chrisfinlay/mri-generative-priors.git
cd mri-generative-priors
pixi install -e gpu
pixi run server-setup      # links data/processed to the shared data, copies checkpoints
pixi run -e gpu check      # should print a CudaDevice
pixi run -e gpu lab
```

Skip `pixi run download` and `pixi run preprocess`. The data is already prepared.
`server-setup` replaces your `data/processed` with a link to the shared,
read-only copy. It also copies the mentor's checkpoints into `checkpoints/`.
They are copied, not linked, so a training run writes to your own file.

## The three splits

| split   | source                               | use it for                        |
|---------|--------------------------------------|-----------------------------------|
| `train` | fastMRI `singlecoil_train` (~90%)    | training priors, fitting spectra  |
| `val`   | fastMRI `singlecoil_train` (~10%)    | tuning β, latent dim, λ, steps…   |
| `test`  | fastMRI `singlecoil_val` (all)       | final numbers, reported once      |

`train` and `val` are split **by patient**, so no patient is in both. The
volume lists are in `/shared/dataset/processed/splits.json`. Load a split with
`FastMRISlices("data/processed", split="val")`. On the server, `split` must be
given, so a script can't silently mix test slices into training.

On a laptop there is only enough data for two splits. There, `val` and `test`
both mean the held-out volumes in `mrigen.data.HELDOUT_VOLUMES`.

## Mentor: once, before the school

The raw volumes go in `/shared/dataset/singlecoil_{train,val}/*.h5`.
`singlecoil_test` is not used: it has no `reconstruction_esc` target. Then:

```bash
pixi run server-prepare    # ~1200 volumes -> /shared/dataset/processed/{train,val,test}
```

It runs in parallel (`--workers`) and can be resumed. Re-running it keeps the
same split and only processes volumes that are missing. The output is made
group-readable but not group-writable.

Put trained weights in `/shared/checkpoints/*.eqx`. Train them on the server
split so the test set stays unseen. Run `pixi run server-setup` in your own
clone first, so `data/processed` points at the shared data:

```bash
pixi run -e gpu python -m mrigen.train_vae --split train --out checkpoints/vae_128.eqx
cp checkpoints/vae_128.eqx /shared/checkpoints/
```

Paths are under `/shared` by default. Set `MRIGEN_SHARED` to use another root.
