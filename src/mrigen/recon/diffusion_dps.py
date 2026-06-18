"""[STRETCH] Diffusion posterior sampling (DPS) for reconstruction.

Optional stretch goal. Alternates an unconditional reverse-diffusion step with a
data-consistency gradient step toward the measured k-space. Requires a trained
score model (models/diffusion.py). See notebooks/05_diffusion_stretch.ipynb.
"""

from __future__ import annotations


def dps_reconstruct(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("Diffusion posterior sampling is a stretch goal")
