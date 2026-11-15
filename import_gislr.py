"""Import signs from Google's Isolated Sign Language Recognition dataset (Kaggle "asl-signs").

The dataset is 250 ASL signs recorded by 21 Deaf signers, already converted to
MediaPipe landmarks. This script downloads ONLY the signs Hermes needs (not the
whole 40 GB), converts each clip into the same 15-frame window features that
collect.py produces, and saves them into data/samples/<LABEL>/ so train.py works
unchanged. Your own recordings in data/samples stay and get trained together.

Setup (once):
  1. Sign in at kaggle.com, open https://www.kaggle.com/competitions/asl-signs/rules
     and click "I Understand and Accept".
  2. kaggle.com -> Settings -> API -> "Create New Token". Then EITHER
       - new-style token:  mkdir -p ~/.kaggle && pbpaste > ~/.kaggle/access_token
         (filename must be literally access_token — NOT a file named KGAT_….
          Copy the token text first; or: export KAGGLE_API_TOKEN=<token>)
       - legacy kaggle.json: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  3. pip install kaggle kagglesdk pandas pyarrow
     (same Python that will run this script: python -m pip install ...)

Run (from ~/Hermes):
  python import_gislr.py                 # default: 60 clips per sign
  python import_gislr.py --per-sign 100  # more data, slower download
  python import_gislr.py --list          # just show which signs exist
"""

from __future__ import annotations

import argparse
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import config
from landmarks import normalize_hand, window_features

COMPETITION = "asl-signs"
CACHE = config.DATA_DIR / "gislr_cache"
AUTH_HELP = (
    "Kaggle login failed.\n"
    "Save your API token as the file ~/.kaggle/access_token\n"
    "(filename must be literally 'access_token'; file contents = the KGAT_… token).\n"
    "Do NOT name the file after the token itself.\n"
    "Or: export KAGGLE_API_TOKEN='KGAT_…'\n"
    "Or legacy: ~/.kaggle/kaggle.json"
)

# Dataset sign name -> Hermes label. Signs missing from the dataset are skipped
# with a warning. I, MEDICINE and PHARMACY are not in this dataset; your own
# recordings of those still get used.
SIGN_MAP = {
    "hello": "HELLO",
    "thankyou": "THANK-YOU",
    "please": "PLEASE",
    "yes": "YES",
    "no": "NO",
    "where": "WHERE",
    "head": "HEAD",
    "owie": "PAIN",          # "it hurts"
    "sick": "SICK",
    "potty": "BATHROOM",
    "water": "WATER",
    "hungry": "HUNGRY",
    "thirsty": "THIRSTY",
    "haveto": "NEED",        # "have to / must"
    "wait": "WAIT",
    "finish": "FINISH",
    "bye": "BYE",
    "fine": "FINE",
    "you": "YOU",
    "minemy": "MY",
    "home": "HOME",
    "help": "HELP",
    "time": "TIME",
    "now": "NOW",
}

WINDOW = config.WINDOW_SIZE
STRIDE = 5
MAX_WINDOWS_PER_CLIP = 4
# Dataset videos are portrait (3:4); the laptop camera is 640x480 (4:3).
# Rescale x so hand shapes have the same proportions as live camera landmarks.
ASPECT_FIX = (config.FRAME_HEIGHT / config.FRAME_WIDTH) / (4 / 3)


def _api():
    """Return an authenticated Kaggle API client.

    Prefer the module-level `kaggle.api` (already authenticated on import in
    kaggle>=1.7). Fall back to constructing KaggleApi. Surface the real error
    instead of a misleading 'pip install' message.
    """
    try:
        import kaggle
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import kaggle ({exc}).\n"
            "Install into THIS interpreter:\n"
            "  python -m pip install -U kaggle kagglesdk pandas pyarrow\n"
            f"  python -c \"import sys; print(sys.executable)\""
        ) from exc

    api = getattr(kaggle, "api", None)
    if api is not None and getattr(api, "_authenticated", False):
        return api

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import KaggleApi ({exc}).\n"
            "Try: python -m pip uninstall -y kaggle kagglesdk && "
            "python -m pip install -U kaggle kagglesdk pandas pyarrow"
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit as exc:
        raise SystemExit(AUTH_HELP) from exc
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{AUTH_HELP}\nUnderlying error: {exc}") from exc
    if not getattr(api, "_authenticated", True):
        raise SystemExit(AUTH_HELP)
    return api


def _download(api, remote: str) -> Path:
    """Download one competition file into the cache (skips if already there)."""
    local = CACHE / remote
    if local.exists() and local.stat().st_size > 0:
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    api.competition_download_file(COMPETITION, remote, path=str(local.parent), quiet=True)
    zipped = local.parent / (local.name + ".zip")
    if zipped.exists():
        with zipfile.ZipFile(zipped) as z:
            z.extractall(local.parent)
        zipped.unlink()
    if not local.exists():
        raise FileNotFoundError(remote)
    return local


def _frame_vectors(parquet: Path) -> list[np.ndarray]:
    """Clip -> list of 126-dim per-frame vectors (left hand, right hand)."""
    df = pd.read_parquet(parquet, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    df = df[df["type"].isin(["left_hand", "right_hand"])]
    vectors = []
    for _, fdf in df.groupby("frame", sort=True):
        vec = np.zeros((2, config.LANDMARKS_PER_HAND, 3), dtype=np.float32)
        has_hand = False
        for slot, hand in enumerate(("left_hand", "right_hand")):
            h = fdf[fdf["type"] == hand].sort_values("landmark_index")
            pts = h[["x", "y", "z"]].to_numpy(dtype=np.float32).copy()
            if len(pts) != config.LANDMARKS_PER_HAND or np.isnan(pts).any():
                continue
            pts[:, 0] *= ASPECT_FIX
            vec[slot] = normalize_hand(pts)
            has_hand = True
        if has_hand:
            vectors.append(vec.reshape(-1))
    return vectors


def _windows(vectors: list[np.ndarray]) -> list[list[np.ndarray]]:
    """Slice a clip into 15-frame windows like the live recognizer sees."""
    if not vectors:
        return []
    if len(vectors) < WINDOW:
        return [vectors + [vectors[-1]] * (WINDOW - len(vectors))]
    starts = list(range(0, len(vectors) - WINDOW + 1, STRIDE))
    if len(starts) > MAX_WINDOWS_PER_CLIP:  # spread evenly across the sign
        idx = np.linspace(0, len(starts) - 1, MAX_WINDOWS_PER_CLIP).round().astype(int)
        starts = [starts[i] for i in idx]
    return [vectors[s:s + WINDOW] for s in starts]


def _mirror(window: list[np.ndarray]) -> list[np.ndarray]:
    """Left/right flip: swap hand slots and negate x. Covers left-handed signers
    and any mirroring difference between the dataset and the live camera."""
    out = []
    for v in window:
        hands = v.reshape(2, config.LANDMARKS_PER_HAND, 3)[::-1].copy()
        hands[..., 0] *= -1
        out.append(hands.reshape(-1))
    return out


def _process(api, row) -> tuple[str, list[np.ndarray]]:
    path = _download(api, row.path)
    feats = []
    for w in _windows(_frame_vectors(path)):
        feats.append(window_features(w))
        feats.append(window_features(_mirror(w)))
    return SIGN_MAP[row.sign], feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sign", type=int, default=60, help="clips per sign to download")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--list", action="store_true", help="list available signs and exit")
    args = ap.parse_args()

    api = _api()
    print("Downloading train.csv ...")
    meta = pd.read_csv(_download(api, "train.csv"))

    available = set(meta["sign"].unique())
    if args.list:
        print(", ".join(sorted(available)))
        return
    missing = [s for s in SIGN_MAP if s not in available]
    if missing:
        print(f"Not in dataset, skipping: {', '.join(missing)}")

    random.seed(0)
    picks = []
    for sign in SIGN_MAP:
        rows = meta[meta["sign"] == sign]
        if rows.empty:
            continue
        # spread picks across different signers so the model generalizes
        rows = rows.sample(frac=1, random_state=0).groupby("participant_id").head(
            max(1, args.per_sign // max(1, rows["participant_id"].nunique()) + 1)
        )
        picks.extend(rows.head(args.per_sign).itertuples())

    print(f"Downloading and converting {len(picks)} clips for "
          f"{len({p.sign for p in picks})} signs ...")
    counts: dict[str, int] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process, api, r) for r in picks]
        for fut in as_completed(futures):
            done += 1
            try:
                label, feats = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  skipped one clip: {exc}")
                continue
            out_dir = config.SAMPLES_DIR / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in feats:
                n = counts.get(label, 0)
                np.save(out_dir / f"gislr_{n:05d}.npy", f)
                counts[label] = n + 1
            if done % 50 == 0 or done == len(picks):
                print(f"  {done}/{len(picks)} clips")

    print("\nImported samples per label:")
    for label, n in sorted(counts.items()):
        print(f"  {label:<10} {n}")
    print("\nNext: record extra '_none' samples, then run: python train.py")


if __name__ == "__main__":
    main()
