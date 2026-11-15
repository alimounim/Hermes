"""Import signs from Google's Isolated Sign Language Recognition dataset (Kaggle "asl-signs").

The dataset is 250 ASL signs recorded by 21 Deaf signers, already converted to
MediaPipe landmarks. This script prefers ONE bulk competition zip download
(avoids hundreds of per-parquet API calls that trigger 429 rate limits), then
converts only the signs Hermes needs into the same 15-frame window features that
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
  3. pip install kaggle kagglesdk six pandas pyarrow
     (same Python that will run this script: python -m pip install ...)

Run (from ~/Hermes):
  python import_gislr.py                           # bulk zip + 60 clips/sign, 1 worker
  python import_gislr.py --workers 1 --per-sign 30 # smaller first import
  python import_gislr.py --per-file                # legacy per-parquet API (rate-limit prone)
  python import_gislr.py --list                    # just show which signs exist
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import config
from landmarks import normalize_hand, window_features

COMPETITION = "asl-signs"
CACHE = config.DATA_DIR / "gislr_cache"
BULK_MARKER = CACHE / ".bulk_download_done"
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

# Serialize Kaggle HTTP downloads — parallel workers still hit 429 hard.
_DOWNLOAD_LOCK = threading.Lock()
_DOWNLOAD_ATTEMPTS = 8


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
            f"Cannot import kaggle: {type(exc).__name__}: {exc}\n"
            "Install into THIS interpreter:\n"
            "  python -m pip install -U kaggle kagglesdk six pandas pyarrow\n"
            "  python -c \"import sys; print(sys.executable)\""
        ) from exc

    api = getattr(kaggle, "api", None)
    if api is not None and getattr(api, "_authenticated", False):
        return api

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import KaggleApi: {type(exc).__name__}: {exc}\n"
            "Try: python -m pip uninstall -y kaggle kagglesdk && "
            "python -m pip install -U kaggle kagglesdk six pandas pyarrow"
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


def _is_retryable(exc: BaseException) -> bool:
    """True for rate limits / transient network failures worth backing off on."""
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    markers = (
        "429",
        "too many requests",
        "rate limit",
        "ratelimit",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "name or service not known",
        "nodename nor servname",
        "temporary failure in name resolution",
        "failed to resolve",
        "dns",
        "reset by peer",
        "broken pipe",
        "503",
        "502",
        "504",
        "ssl",
    )
    return any(m in msg or m in name for m in markers)


def _resolve_cached(remote: str) -> Path | None:
    """Find a competition-relative path already present under the cache."""
    candidates = [
        CACHE / remote,
        CACHE / COMPETITION / remote,
        CACHE / Path(remote).name,
    ]
    # After bulk unzip, train.csv sometimes lands at cache root while parquets
    # keep their train_landmark_files/... layout (or vice versa).
    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    matches = list(CACHE.rglob(Path(remote).name))
    for cand in matches:
        if cand.is_file() and cand.stat().st_size > 0:
            # Prefer a path that ends with the remote relative path.
            try:
                if cand.relative_to(CACHE).as_posix().endswith(remote):
                    return cand
            except ValueError:
                pass
    return matches[0] if matches else None


def _unzip_if_needed(path: Path, remote: str) -> Path | None:
    """If Kaggle wrapped the file in a .zip, extract and return the payload path."""
    zipped = path if path.suffix == ".zip" else path.parent / (path.name + ".zip")
    # competition_download_file may write basename.zip into path= parent.
    basename_zip = path.parent / (Path(remote).name + ".zip")
    for zpath in (zipped, basename_zip):
        if not zpath.exists():
            continue
        with zipfile.ZipFile(zpath) as z:
            z.extractall(zpath.parent)
        zpath.unlink(missing_ok=True)
        found = _resolve_cached(remote)
        if found:
            return found
    return _resolve_cached(remote)


def _download_file(api, remote: str, attempts: int = _DOWNLOAD_ATTEMPTS) -> Path:
    """Download one competition file into the cache (skips if already there)."""
    found = _resolve_cached(remote)
    if found:
        return found
    dest_parent = (CACHE / remote).parent
    dest_parent.mkdir(parents=True, exist_ok=True)
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            with _DOWNLOAD_LOCK:
                found = _resolve_cached(remote)
                if found:
                    return found
                api.competition_download_file(
                    COMPETITION,
                    remote,
                    path=str(dest_parent),
                    quiet=True,
                    unzip=True,
                )
                time.sleep(0.35)
            found = _unzip_if_needed(CACHE / remote, remote)
            if found:
                return found
            raise FileNotFoundError(
                f"downloaded '{remote}' but file not found under {CACHE}"
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 >= attempts or not _is_retryable(exc):
                raise
            delay = min(60.0, (2**attempt) + random.random())
            print(
                f"  retry {attempt + 1}/{attempts} {remote}: "
                f"{type(exc).__name__}: {exc} (sleep {delay:.1f}s)"
            )
            time.sleep(delay)
    raise RuntimeError(f"download failed: {remote}") from last_exc


def _bulk_ready() -> bool:
    if BULK_MARKER.exists():
        return True
    # Heuristic: landmark tree already present from a prior unzip.
    return (CACHE / "train_landmark_files").is_dir() or (
        CACHE / COMPETITION / "train_landmark_files"
    ).is_dir()


def _download_bulk(api, attempts: int = _DOWNLOAD_ATTEMPTS) -> None:
    """One competition zip via competition_download_files (avoids per-file 429s).

    Warning: the asl-signs archive is large (~tens of GB). Prefer this over
    thousands of per-parquet API calls when rate-limited.
    """
    if _bulk_ready():
        print(f"Using cached competition data under {CACHE}")
        BULK_MARKER.touch()
        return

    CACHE.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading full '{COMPETITION}' competition zip into {CACHE} "
        "(large; one API download — avoids 429 from per-parquet calls) ..."
    )
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            with _DOWNLOAD_LOCK:
                api.competition_download_files(
                    COMPETITION,
                    path=str(CACHE),
                    force=False,
                    quiet=False,
                    unzip=True,
                )
            # Some Kaggle versions leave the zip behind even with unzip=True.
            for zpath in CACHE.glob("*.zip"):
                print(f"  extracting {zpath.name} ...")
                with zipfile.ZipFile(zpath) as z:
                    z.extractall(CACHE)
                zpath.unlink(missing_ok=True)
            if not _bulk_ready() and not (CACHE / "train.csv").exists():
                raise FileNotFoundError(
                    f"bulk download finished but train.csv / landmark files "
                    f"not found under {CACHE}"
                )
            BULK_MARKER.touch()
            print("Bulk download complete.")
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 >= attempts or not _is_retryable(exc):
                raise
            delay = min(90.0, (2**attempt) + random.random())
            print(
                f"  bulk retry {attempt + 1}/{attempts}: "
                f"{type(exc).__name__}: {exc} (sleep {delay:.1f}s)"
            )
            time.sleep(delay)
    raise RuntimeError("bulk competition download failed") from last_exc


def _download(api, remote: str, *, per_file: bool) -> Path:
    """Resolve a competition file from bulk cache, else per-file download."""
    found = _resolve_cached(remote)
    if found:
        return found
    if not per_file:
        raise FileNotFoundError(
            f"missing '{remote}' under {CACHE} after bulk download "
            f"(tried cache roots and rglob by basename)"
        )
    return _download_file(api, remote)


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


def _process(api, row, *, per_file: bool) -> tuple[str, list[np.ndarray]]:
    path = _download(api, row.path, per_file=per_file)
    feats = []
    for w in _windows(_frame_vectors(path)):
        feats.append(window_features(w))
        feats.append(window_features(_mirror(w)))
    return SIGN_MAP[row.sign], feats


def _save_feats(counts: dict[str, int], label: str, feats: list[np.ndarray]) -> None:
    out_dir = config.SAMPLES_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in feats:
        n = counts.get(label, 0)
        np.save(out_dir / f"gislr_{n:05d}.npy", f)
        counts[label] = n + 1


def _format_skip(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sign", type=int, default=60, help="clips per sign to download")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel clip workers (default 1; keep low to avoid 429s)",
    )
    ap.add_argument(
        "--per-file",
        action="store_true",
        help="download each parquet via the API (rate-limit prone); "
        "default is one bulk competition zip",
    )
    ap.add_argument("--list", action="store_true", help="list available signs and exit")
    args = ap.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    api = _api()

    if args.per_file:
        print("Downloading train.csv (per-file mode) ...")
        meta = pd.read_csv(_download_file(api, "train.csv"))
    else:
        _download_bulk(api)
        train_csv = _resolve_cached("train.csv")
        if train_csv is None:
            print("train.csv missing after bulk download; fetching just that file ...")
            train_csv = _download_file(api, "train.csv")
        print(f"Reading {train_csv} ...")
        meta = pd.read_csv(train_csv)

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

    mode = "per-file API" if args.per_file else "bulk zip cache"
    print(
        f"Converting {len(picks)} clips for "
        f"{len({p.sign for p in picks})} signs "
        f"(workers={args.workers}, mode={mode}) ..."
    )
    counts: dict[str, int] = {}
    done = 0
    skipped = 0

    def _handle_result(label: str, feats: list[np.ndarray]) -> None:
        _save_feats(counts, label, feats)

    if args.workers == 1:
        for row in picks:
            done += 1
            try:
                label, feats = _process(api, row, per_file=args.per_file)
                _handle_result(label, feats)
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"  skipped one clip: {_format_skip(exc)}")
            if done % 25 == 0 or done == len(picks):
                print(f"  {done}/{len(picks)} clips ({sum(counts.values())} samples)")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(_process, api, r, per_file=args.per_file) for r in picks
            ]
            for fut in as_completed(futures):
                done += 1
                try:
                    label, feats = fut.result()
                    _handle_result(label, feats)
                except Exception as exc:  # noqa: BLE001
                    skipped += 1
                    print(f"  skipped one clip: {_format_skip(exc)}")
                if done % 25 == 0 or done == len(picks):
                    print(f"  {done}/{len(picks)} clips ({sum(counts.values())} samples)")

    total = sum(counts.values())
    print("\nImported samples per label:")
    if not counts:
        print("  (none)")
    else:
        for label, n in sorted(counts.items()):
            print(f"  {label:<10} {n}")
    print(f"Total samples: {total}  (skipped clips: {skipped})")

    if total == 0:
        print(
            "\nERROR: imported 0 samples. Likely Kaggle 429 / network, or bulk zip "
            "missing landmark files.\n"
            "Wait a few minutes, then:\n"
            "  python import_gislr.py --workers 1 --per-sign 30\n"
            "If bulk zip is too large, free disk space under data/gislr_cache/ and retry.\n"
            "Avoid --per-file unless you must (it hammers the API).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("\nNext: record extra '_none' samples, then run: python train.py")


if __name__ == "__main__":
    main()
