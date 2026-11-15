"""Extract hand + upper-body landmarks for every GISLR clip of the Hermes signs.

Reads the unpacked Kaggle "asl-signs" folder (data/asl-signs/) and writes one
compact file, data/gislr_clips.npz, that train.py learns from. Unlike the old
import this keeps whole clips (one clip = one sign) and the body landmarks, so
the model can see WHERE the hands are, not only their shape.

  python build_dataset.py                 # all clips of the mapped signs
  python build_dataset.py --src /path/to/asl-signs
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import config
from import_gislr import SIGN_MAP

# Row layout of every GISLR parquet frame: face 468, left hand 21, pose 33, right hand 21.
ROWS_PER_FRAME = 543
LH = slice(468, 489)
POSE = slice(489, 522)
RH = slice(522, 543)

OUT = config.DATA_DIR / "gislr_clips.npz"
# Clips of signs outside the vocabulary, used as an "_other" class so unknown
# movements can be rejected instead of forced into the nearest known sign.
OTHER_PER_SIGN = 12


def load_clip(path: str):
    """Return (left_hand, right_hand, pose) as float32 arrays, NaN where missing."""
    df = pd.read_parquet(path, columns=["x", "y", "z"])
    arr = df.to_numpy(dtype=np.float32)
    if len(arr) % ROWS_PER_FRAME:
        return None
    arr = arr.reshape(-1, ROWS_PER_FRAME, 3)
    return arr[:, LH], arr[:, RH], arr[:, POSE][:, config.POSE_KEEP, :2]


def _load(args):
    path, = args
    try:
        return load_clip(path)
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=config.DATA_DIR / "asl-signs")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    meta = pd.read_csv(args.src / "train.csv")
    known = meta[meta["sign"].isin(SIGN_MAP)].copy()
    known["label"] = known["sign"].map(SIGN_MAP)
    other = (
        meta[~meta["sign"].isin(SIGN_MAP)]
        .groupby("sign", group_keys=False)
        .sample(n=OTHER_PER_SIGN, random_state=0)
        .copy()
    )
    other["label"] = "_other"
    rows = pd.concat([known, other], ignore_index=True)
    print(f"{len(known)} clips of {known['label'].nunique()} signs + {len(other)} '_other' clips")

    paths = [(str(args.src / p),) for p in rows["path"]]
    with Pool(args.workers) as pool:
        clips = pool.map(_load, paths, chunksize=64)

    keep = [i for i, c in enumerate(clips) if c is not None]
    lh = np.concatenate([clips[i][0] for i in keep]).astype(np.float16)
    rh = np.concatenate([clips[i][1] for i in keep]).astype(np.float16)
    pose = np.concatenate([clips[i][2] for i in keep]).astype(np.float16)
    lengths = np.array([len(clips[i][0]) for i in keep], dtype=np.int32)
    rows = rows.iloc[keep]
    np.savez_compressed(
        OUT,
        lh=lh, rh=rh, pose=pose, lengths=lengths,
        labels=rows["label"].to_numpy(dtype=str),
        participants=rows["participant_id"].to_numpy(dtype=np.int64),
    )
    print(f"Saved {len(keep)} clips ({lengths.sum()} frames) → {OUT}")


if __name__ == "__main__":
    main()
