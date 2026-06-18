# Register for fastMRI data — do this first

fastMRI is **patient-derived data**. You may not use it until you have
registered and agreed to the NYU fastMRI **Data Sharing Agreement (DSA)**, and
you may **not** share the data or your download links with anyone — including
your teammates. Each person downloads their own copy.

## Steps

1. Go to <https://fastmri.med.nyu.edu/> and request access.
2. Read and accept the Data Sharing Agreement.
3. You will receive an email with **personal, time-limited download links** for
   each archive (curl/wget URLs).
4. We only need the **single-coil validation** set
   (`knee_singlecoil_val`) — a handful of volumes is plenty for a 128² VAE.
   You do **not** need the 88 GB training set.
5. Copy your `singlecoil_val` link into an environment variable and run the
   downloader:

   ```bash
   export FASTMRI_VAL_URL="https://<your-personal-link>/knee_singlecoil_val.tar.gz"
   pixi run download        # -> data/raw/*.h5
   pixi run preprocess      # -> data/processed/*.npz  (128x128 magnitude slices)
   ```

## What this repo does and does not contain

- **No raw or derived patient data** is committed to this repository.
- `download_subset.py` reads your personal link from `$FASTMRI_VAL_URL`; it
  never bundles data or hard-codes a link.
- `data/raw/` and `data/processed/` are git-ignored.
- The provided model checkpoints are **trained weights, not data**, and are safe
  to redistribute.

If your link expires, just request a fresh one from the fastMRI site.
