"""Download a subset of fastMRI single-coil validation volumes.

Reads a **per-user** download link from the FASTMRI_VAL_URL environment
variable (see data/REGISTER_FIRST.md). This script never bundles data and never
hard-codes a link, per the Data Sharing Agreement.

    export FASTMRI_VAL_URL="https://<your-personal-link>/knee_singlecoil_val.tar.gz"
    python data/download_subset.py --n 5
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
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
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "singlecoil_val.tar.gz"
        print(f"downloading archive (this may take a while)...")
        urllib.request.urlretrieve(url, archive)

        print("extracting first", n, "volumes...")
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".h5")]
            members.sort(key=lambda m: m.name)
            for m in members[:n]:
                m.name = Path(m.name).name  # flatten into raw/
                tar.extract(m, RAW_DIR)
                print("  ->", RAW_DIR / m.name)

    kept = sorted(RAW_DIR.glob("*.h5"))
    print(f"done: {len(kept)} volume(s) in {RAW_DIR}")
    print("next: python data/preprocess.py")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=5, help="number of volumes to keep")
    download(p.parse_args().n)


if __name__ == "__main__":
    main()
