"""School GPU server only: shared data and checkpoints for every student clone.

On the server the fastMRI volumes are downloaded once into a shared directory,
not per student. Two steps:

    pixi run server-prepare   # mentor, once: .h5 -> processed/{train,val,test}
    pixi run server-setup     # each student, in their own clone

Layout under ``$MRIGEN_SHARED`` (default ``/shared``)::

    dataset/singlecoil_train/*.h5   -> split by patient into train / val
    dataset/singlecoil_val/*.h5     -> test (the final, untouched test set)
    dataset/processed/{train,val,test}/*.npz, splits.json   (written by prepare)
    checkpoints/*.eqx               (mentor drops trained weights here)

``singlecoil_test`` is not used: it has no ``reconstruction_esc`` target.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "data"))
from preprocess import preprocess_volume

SHARED = Path(os.environ.get("MRIGEN_SHARED", "/shared"))
DATASET = SHARED / "dataset"
PROCESSED = DATASET / "processed"
CHECKPOINTS = SHARED / "checkpoints"

VAL_FRACTION = 0.1  # of singlecoil_train volumes, held back for tuning
SEED = 0


def split_by_patient(vols: list[Path], val_fraction: float, seed: int) -> tuple[list, list]:
    """Split volumes into (train, val) so no patient appears in both."""
    by_patient: dict[str, list[Path]] = {}
    for v in vols:
        with h5py.File(v, "r") as f:
            by_patient.setdefault(str(f.attrs.get("patient_id", v.stem)), []).append(v)
    patients = sorted(by_patient)
    np.random.default_rng(seed).shuffle(patients)
    val, n_val = [], round(val_fraction * len(vols))
    for pid in patients:
        if len(val) >= n_val:
            break
        val += by_patient[pid]
    val_set = set(val)
    return sorted(v for v in vols if v not in val_set), sorted(val)


def _work(args: tuple[Path, Path, int]) -> int:
    vol, out_dir, size = args
    return preprocess_volume(vol, out_dir, size)


def prepare(size: int, workers: int, force: bool) -> None:
    """Preprocess the shared raw volumes into processed/{train,val,test}."""
    train_vols = sorted((DATASET / "singlecoil_train").glob("*.h5"))
    test_vols = sorted((DATASET / "singlecoil_val").glob("*.h5"))
    if not train_vols or not test_vols:
        raise SystemExit(f"Expected .h5 files in {DATASET}/singlecoil_train and singlecoil_val.")

    manifest = PROCESSED / "splits.json"
    if manifest.exists():  # keep the split stable across re-runs
        names = json.loads(manifest.read_text())
        train = [v for v in train_vols if v.stem in set(names["train"])]
        val = [v for v in train_vols if v.stem in set(names["val"])]
    else:
        print(f"splitting {len(train_vols)} singlecoil_train volumes by patient ...")
        train, val = split_by_patient(train_vols, VAL_FRACTION, SEED)
    splits = {"train": train, "val": val, "test": test_vols}

    jobs = []
    for split, vols in splits.items():
        out = PROCESSED / split
        out.mkdir(parents=True, exist_ok=True)
        jobs += [(v, out, size) for v in vols if force or not (out / f"{v.stem}.npz").exists()]
    print(f"{len(jobs)} volume(s) to preprocess with {workers} worker(s)")
    with ProcessPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(_work, jobs, chunksize=4), 1):
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")

    manifest.write_text(
        json.dumps(
            {
                "source": {
                    "train": "singlecoil_train",
                    "val": "singlecoil_train",
                    "test": "singlecoil_val",
                },
                "val_fraction": VAL_FRACTION,
                "seed": SEED,
                **{k: [v.stem for v in vols] for k, vols in splits.items()},
            },
            indent=1,
        )
    )
    # group may read, not write: one student can't break the data for everyone
    for path in [PROCESSED, *PROCESSED.rglob("*")]:
        path.chmod(0o750 if path.is_dir() else 0o640)
    for split, vols in splits.items():
        print(f"{split:>5}: {len(vols)} volumes in {PROCESSED / split}")


def setup() -> None:
    """Point this clone's data/processed at the shared data; copy checkpoints."""
    if not (PROCESSED / "splits.json").exists():
        raise SystemExit(
            f"No prepared data at {PROCESSED}. Ask the mentor to run `pixi run server-prepare`."
        )

    link = REPO / "data" / "processed"
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        if any(link.iterdir()):
            raise SystemExit(f"{link} exists and is not empty; move it aside and re-run.")
        link.rmdir()
    link.symlink_to(PROCESSED, target_is_directory=True)
    print(f"data/processed -> {PROCESSED}")

    # Copied, not linked: training with the default --out would otherwise
    # write through the link onto everyone's shared checkpoint.
    ckpt_dir = REPO / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    for src in sorted(CHECKPOINTS.glob("*.eqx")):
        dst = ckpt_dir / src.name
        if dst.exists():
            print(f"checkpoints/{src.name}: already present, kept yours")
        else:
            shutil.copy2(src, dst)
            print(f"checkpoints/{src.name}: copied from {CHECKPOINTS}")

    names = json.loads((PROCESSED / "splits.json").read_text())
    print("splits: " + ", ".join(f"{s} {len(names[s])} volumes" for s in ("train", "val", "test")))
    print("Tune on split='val'; report final numbers on split='test' once.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare", help="mentor, once: preprocess the shared raw data")
    pp.add_argument("--size", type=int, default=128)
    pp.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    pp.add_argument("--force", action="store_true", help="redo volumes already processed")
    sub.add_parser("setup", help="each student: link this clone to the shared data")
    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args.size, args.workers, args.force)
    else:
        setup()


if __name__ == "__main__":
    main()
