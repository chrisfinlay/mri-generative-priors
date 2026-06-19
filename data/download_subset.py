"""Download a subset of fastMRI single-coil validation volumes.

Reads a **per-user** download link from the FASTMRI_VAL_URL environment
variable (see data/REGISTER_FIRST.md). This script never bundles data and never
hard-codes a link, per the Data Sharing Agreement.

    export FASTMRI_VAL_URL="https://<your-personal-link>/knee_singlecoil_val.tar.xz"
    python data/download_subset.py --n 5

It **streams** the archive and stops after extracting ``n`` volumes, so you only
transfer/decompress as much as you need — not the whole 15 GB. Works for both
``.tar.gz`` and ``.tar.xz`` (compression is auto-detected).
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"


def download(n: int) -> None:
    url = os.environ.get("FASTMRI_VAL_URL")
    if not url:
        sys.exit(
            "FASTMRI_VAL_URL is not set. Register at https://fastmri.med.nyu.edu/, "
            "accept the Data Sharing Agreement, then:\n"
            '  export FASTMRI_VAL_URL="<your personal singlecoil_val link>"\n'
            "See data/REGISTER_FIRST.md."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"streaming archive, keeping the first {n} volume(s)...")

    kept = 0
    # Stream (mode "r|*"): read the archive sequentially from the network and
    # stop early — no full download, no full decompression.
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (user-supplied DSA link)
        with tarfile.open(fileobj=resp, mode="r|*") as tar:
            for member in tar:
                if not member.name.endswith(".h5"):
                    continue
                member.name = Path(member.name).name  # flatten into raw/
                tar.extract(member, RAW_DIR, filter="data")
                kept += 1
                print("  ->", RAW_DIR / member.name)
                if kept >= n:
                    break

    print(f"done: {kept} volume(s) in {RAW_DIR}")
    print("next: python data/preprocess.py")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=5, help="number of volumes to keep")
    download(p.parse_args().n)


if __name__ == "__main__":
    main()
