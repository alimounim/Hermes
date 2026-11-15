"""Import signs from Google's Isolated Sign Language Recognition dataset (Kaggle "asl-signs").

The dataset is 250 ASL signs recorded by 21 Deaf signers, already converted to
MediaPipe landmarks. This script downloads train.csv plus only the parquet
clips it selects (~1.5MB each, sequentially with backoff on 429s), then
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
  python import_gislr.py --per-sign 30     # fetch ~700 clips (~1GB), import
  python import_gislr.py                   # 60 clips/sign
  python import_gislr.py --bulk            # full ~40GB competition zip instead
  python import_gislr.py --list            # just show which signs exist

Re-running is safe: clips already in data/gislr_cache are reused, and files
left in the wrong place by older versions of this script are repaired.
"""

from __future__ import annotations

import argparse
import random
import shutil
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
# Full asl-signs train set has ~94k parquet clips. Partial/corrupt extracts are
# typically a few hundred files and cause mass FileNotFoundError at convert time.
MIN_PARQUET_FILES = 5_000
MIN_PICK_COVERAGE = 0.9
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
    # Needs
    "drink": "DRINK",
    "food": "FOOD",
    "sleep": "SLEEP",
    "sleepy": "SLEEPY",
    "hot": "HOT",
    "bath": "BATH",
    "bed": "BED",
    # Feelings
    "bad": "BAD",
    "better": "BETTER",
    "happy": "HAPPY",
    "sad": "SAD",
    "cry": "CRY",
    # Questions
    "who": "WHO",
    "why": "WHY",
    # Actions
    "go": "GO",
    "stay": "STAY",
    "give": "GIVE",
    "have": "HAVE",
    "look": "LOOK",
    "listen": "LISTEN",
    "find": "FIND",
    # Time
    "later": "LATER",
    "tomorrow": "TOMORROW",
    "yesterday": "YESTERDAY",
    "morning": "MORNING",
    "night": "NIGHT",
    # Negation / quantity
    "not": "NOT",
    "another": "ANOTHER",
    # People
    "mom": "MOM",
    "dad": "DAD",
    "person": "PERSON",
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


def _is_not_found(exc: BaseException) -> bool:
    """True for HTTP 404 / missing competition file (do not retry)."""
    msg = str(exc).lower()
    return (
        "404" in msg
        or "not found" in msg
        or "does not exist" in msg
        or "no such file" in msg
    )


def _is_retryable(exc: BaseException) -> bool:
    """True for rate limits / transient network failures worth backing off on."""
    if _is_not_found(exc):
        return False
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


def _repair_cache() -> None:
    """Fix layouts left by older versions of this script.

    Older code passed path=<cache>/train_landmark_files/<id> to the Kaggle API,
    which appends the full remote path, so files landed at
    train_landmark_files/<id>/train_landmark_files/<id>/<clip>.parquet, and some
    stayed wrapped as <clip>.parquet.zip. Extract those and move every parquet
    to its canonical CACHE/train_landmark_files/<id>/<clip>.parquet location.
    """
    root = CACHE / "train_landmark_files"
    if not root.is_dir():
        return
    extracted = moved = 0
    for zpath in list(root.rglob("*.parquet.zip")):
        try:
            with zipfile.ZipFile(zpath) as z:
                z.extractall(zpath.parent)
            zpath.unlink()
            extracted += 1
        except zipfile.BadZipFile:
            print(f"  removing corrupt {zpath.relative_to(CACHE)}")
            zpath.unlink()
    for pq in list(root.rglob("*.parquet")):
        rel = pq.relative_to(root).parts
        if len(rel) == 2:
            continue  # already <id>/<clip>.parquet
        dest = root / rel[-2] / rel[-1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            pq.unlink()
        else:
            pq.rename(dest)
        moved += 1
    # Drop the now-empty doubled directories.
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    if extracted or moved:
        print(f"Repaired cache: extracted {extracted} zips, relocated {moved} parquets")


def _download_file(api, remote: str, attempts: int = _DOWNLOAD_ATTEMPTS) -> Path:
    """Download one competition file into CACHE/<remote> (skips if already there)."""
    found = _resolve_cached(remote)
    if found:
        return found
    CACHE.mkdir(parents=True, exist_ok=True)
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            with _DOWNLOAD_LOCK:
                found = _resolve_cached(remote)
                if found:
                    return found
                # The API writes to <path>/<remote>, so path must be the cache
                # root — passing the file's parent dir doubles the subpath.
                api.competition_download_file(
                    COMPETITION,
                    remote,
                    path=str(CACHE),
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
            if _is_not_found(exc):
                raise FileNotFoundError(
                    f"404 for '{remote}' via competition_download_file: {exc}"
                ) from exc
            if attempt + 1 >= attempts or not _is_retryable(exc):
                raise
            delay = min(60.0, (2**attempt) + random.random())
            print(
                f"  retry {attempt + 1}/{attempts} {remote}: "
                f"{type(exc).__name__}: {exc} (sleep {delay:.1f}s)"
            )
            time.sleep(delay)
    raise RuntimeError(f"download failed: {remote}") from last_exc


def _landmark_roots() -> list[Path]:
    roots = []
    for cand in (CACHE / "train_landmark_files", CACHE / COMPETITION / "train_landmark_files"):
        if cand.is_dir():
            roots.append(cand)
    return roots


def _count_parquets() -> int:
    return sum(1 for root in _landmark_roots() for _ in root.rglob("*.parquet"))


def _wipe_cache() -> None:
    """Remove data/gislr_cache so the next bulk fetch starts clean."""
    if CACHE.exists():
        print(f"Wiping cache {CACHE} ...")
        shutil.rmtree(CACHE)
    CACHE.mkdir(parents=True, exist_ok=True)


def _pick_coverage(remotes: list[str]) -> tuple[int, int, list[str]]:
    missing: list[str] = []
    found = 0
    for remote in remotes:
        if _resolve_cached(remote):
            found += 1
        else:
            missing.append(remote)
    return found, len(remotes), missing


def _bulk_looks_complete(required: list[str] | None = None) -> bool:
    """True only if train.csv exists and the landmark tree looks fully extracted."""
    if _resolve_cached("train.csv") is None:
        print("Cache incomplete: train.csv missing")
        return False
    if not _landmark_roots():
        print("Cache incomplete: train_landmark_files/ missing")
        return False
    n = _count_parquets()
    if n < MIN_PARQUET_FILES:
        print(
            f"Cache incomplete: only {n} parquet files "
            f"(need >= {MIN_PARQUET_FILES}; full set is ~94k)"
        )
        return False
    if required is not None:
        found, total, missing = _pick_coverage(required)
        if total and found < int(MIN_PICK_COVERAGE * total):
            print(
                f"Cache incomplete for selected clips: {found}/{total} present "
                f"({len(missing)} missing), e.g. {missing[:3]}"
            )
            return False
        print(f"Cache coverage for selected clips: {found}/{total}")
    else:
        print(f"Cache looks complete: {n} parquet files under {CACHE}")
    return True


def _extract_zips_in_cache() -> None:
    """Extract every .zip under CACHE, including nested archives, until none remain."""
    rounds = 0
    while True:
        zips = sorted(p for p in CACHE.rglob("*.zip") if p.is_file())
        if not zips:
            return
        rounds += 1
        if rounds > 20:
            raise RuntimeError(f"too many nested zip rounds under {CACHE}")
        for zpath in zips:
            size_mb = zpath.stat().st_size / (1024 * 1024)
            rel = zpath.relative_to(CACHE)
            # Main competition bundle is ~40GB; tiny zips are OK (metadata), but a
            # supposed asl-signs.zip under ~1GB means a truncated download.
            if "asl-signs" in zpath.name.lower() and size_mb < 1000:
                raise RuntimeError(
                    f"{rel} is only {size_mb:.1f} MB — truncated competition download. "
                    "Need ~50+ GB free disk, then: python import_gislr.py --force-download "
                    "--workers 1 --per-sign 30"
                )
            print(f"  extracting {rel} ({size_mb:.1f} MB) ...")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(zpath.parent)
            zpath.unlink(missing_ok=True)


def _list_top_level_zip_names(api) -> list[str]:
    """Return top-level *.zip names from competition_list_files (paginated)."""
    names: list[str] = []
    page_token = None
    try:
        for _ in range(50):  # safety cap
            resp = api.competition_list_files(
                COMPETITION, page_token=page_token, page_size=200
            )
            files = list(getattr(resp, "files", None) or [])
            for f in files:
                name = getattr(f, "name", None) or getattr(f, "ref", None) or ""
                # Only top-level archives / files — not nested parquet paths.
                if "/" in name or "\\" in name:
                    continue
                if name.lower().endswith(".zip"):
                    names.append(name)
            page_token = (
                getattr(resp, "next_page_token", None)
                or getattr(resp, "nextPageToken", None)
            )
            if not page_token or not files:
                break
    except Exception as exc:  # noqa: BLE001
        print(f"  competition_list_files skipped ({type(exc).__name__}: {exc})")
    return sorted(set(names))


def _download_listed_zips(api) -> None:
    """Download every top-level competition .zip Kaggle lists (in case there are several)."""
    for name in _list_top_level_zip_names(api):
        dest = CACHE / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  already have {name}")
            continue
        print(f"  downloading listed archive {name} ...")
        with _DOWNLOAD_LOCK:
            api.competition_download_file(
                COMPETITION, name, path=str(CACHE), force=True, quiet=False, unzip=False
            )
            # API sometimes writes name.zip.zip
            alt = CACHE / (name + ".zip")
            if not dest.exists() and alt.exists():
                alt.rename(dest)


def _download_bulk(
    api,
    *,
    force: bool = False,
    required: list[str] | None = None,
    attempts: int = _DOWNLOAD_ATTEMPTS,
) -> None:
    """Fetch ALL competition archives, extract every zip, verify completeness.

    Warning: asl-signs is ~40GB compressed / ~50GB+ extracted. Prefer this over
    thousands of per-parquet API calls when rate-limited.
    """
    if force:
        _wipe_cache()
    elif _bulk_looks_complete(required):
        BULK_MARKER.touch()
        return
    else:
        # Stale marker / truncated extract (e.g. 266 parquets / 148MB).
        if CACHE.exists() and any(CACHE.iterdir()):
            print("Existing cache failed completeness checks; wiping before re-fetch.")
            _wipe_cache()

    CACHE.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading full '{COMPETITION}' competition data into {CACHE} "
        "(~40GB+; extracts every zip Kaggle returns) ..."
    )
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            with _DOWNLOAD_LOCK:
                api.competition_download_files(
                    COMPETITION,
                    path=str(CACHE),
                    force=True,
                    quiet=False,
                    unzip=False,  # we extract ALL zips ourselves (incl. nested)
                )
            # Extra top-level zips if the bundle did not include everything.
            _download_listed_zips(api)
            _extract_zips_in_cache()
            if not _bulk_looks_complete(required):
                raise FileNotFoundError(
                    f"bulk download finished but cache still incomplete under {CACHE}. "
                    "Check free disk (~50+ GB), then re-run with --force-download."
                )
            BULK_MARKER.touch()
            print(f"Bulk download complete ({_count_parquets()} parquet files).")
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


def _clear_prior_gislr_samples() -> None:
    """Remove previous gislr_*.npy so a full re-import does not mix with partial runs."""
    removed = 0
    for label in sorted(set(SIGN_MAP.values())):
        d = config.SAMPLES_DIR / label
        if not d.is_dir():
            continue
        for f in d.glob("gislr_*.npy"):
            f.unlink()
            removed += 1
    if removed:
        print(f"Cleared {removed} prior gislr_*.npy samples before re-import")


def _cached(remote: str) -> Path:
    """Path of a competition file that should already be in the cache."""
    found = _resolve_cached(remote)
    if found is None:
        raise FileNotFoundError(f"'{remote}' not in {CACHE} (download failed earlier)")
    return found


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


def _process(row) -> tuple[str, list[np.ndarray]]:
    path = _cached(row.path)
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


def _pick_clips(meta: pd.DataFrame, per_sign: int) -> list:
    """Choose up to per_sign clips per mapped sign, spread across signers."""
    picks = []
    for sign in SIGN_MAP:
        rows = meta[meta["sign"] == sign]
        if rows.empty:
            continue
        rows = rows.sample(frac=1, random_state=0).groupby("participant_id").head(
            max(1, per_sign // max(1, rows["participant_id"].nunique()) + 1)
        )
        picks.extend(rows.head(per_sign).itertuples())
    return picks


def _fetch_missing(api, picks: list) -> None:
    """Targeted mode: download only the selected parquets not already cached."""
    found, total, missing = _pick_coverage([p.path for p in picks])
    print(f"Selected-clip cache coverage: {found}/{total}")
    if not missing:
        return
    print(f"Downloading {len(missing)} missing parquets (one at a time) ...")
    failed = 0
    for i, remote in enumerate(missing, 1):
        try:
            _download_file(api, remote)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  could not fetch {remote}: {_format_skip(exc)}")
        if i % 25 == 0 or i == len(missing):
            print(f"  {i}/{len(missing)} fetched ({failed} failed)")


class _LocalZip:
    """A competition zip you downloaded yourself; extracts only what is asked for."""

    def __init__(self, path: Path):
        if not path.is_file():
            raise SystemExit(f"--zip file not found: {path}")
        try:
            self.zf = zipfile.ZipFile(path)
        except zipfile.BadZipFile as exc:
            raise SystemExit(
                f"{path} is not a complete zip (still downloading?): {exc}"
            ) from exc
        # Map competition-relative path -> member name, tolerating a wrapper
        # folder such as asl-signs/train_landmark_files/...
        self.members: dict[str, str] = {}
        for name in self.zf.namelist():
            for anchor in ("train_landmark_files/", "train.csv"):
                i = name.find(anchor)
                if i != -1:
                    self.members[name[i:]] = name
                    break

    def extract(self, remote: str) -> Path | None:
        member = self.members.get(remote)
        if member is None:
            return None
        dest = CACHE / remote
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self.zf.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        return dest

    def extract_missing(self, remotes: list[str]) -> None:
        todo = [r for r in remotes if not _resolve_cached(r)]
        print(f"Extracting {len(todo)} parquets from {self.zf.filename} ...")
        absent = [r for r in todo if self.extract(r) is None]
        if absent:
            print(f"  {len(absent)} not in the zip, e.g. {absent[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sign", type=int, default=60, help="clips per sign to convert")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel conversion workers (downloads are always sequential)",
    )
    ap.add_argument(
        "--bulk",
        action="store_true",
        help="download the full ~40GB competition zip instead of only the "
        "selected parquets",
    )
    ap.add_argument(
        "--zip",
        type=Path,
        help="use a competition zip you already downloaded (e.g. "
        "~/Downloads/asl-signs.zip); extracts only the selected clips",
    )
    ap.add_argument(
        "--force-download",
        action="store_true",
        help="wipe data/gislr_cache first and re-download everything needed",
    )
    ap.add_argument(
        "--per-file",
        action="store_true",
        help=argparse.SUPPRESS,  # old flag; targeted per-file is now the default
    )
    ap.add_argument("--list", action="store_true", help="list available signs and exit")
    args = ap.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    local_zip = _LocalZip(args.zip.expanduser()) if args.zip else None
    api = None if local_zip else _api()

    if local_zip:
        print(f"Download mode: local zip {args.zip}")
        _repair_cache()
        if not _resolve_cached("train.csv") and not local_zip.extract("train.csv"):
            raise SystemExit(f"train.csv not found in {args.zip}")
    elif args.bulk:
        print("Download mode: full competition zip (competition_download_files)")
        _download_bulk(api, force=args.force_download)
    else:
        print("Download mode: targeted (only the selected parquets)")
        if args.force_download:
            _wipe_cache()
        _repair_cache()

    train_csv = _resolve_cached("train.csv") or _download_file(api, "train.csv")
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
    picks = _pick_clips(meta, args.per_sign)

    if local_zip:
        local_zip.extract_missing([p.path for p in picks])
    elif args.bulk:
        found, total_req, missing_paths = _pick_coverage([p.path for p in picks])
        print(f"Selected-clip cache coverage: {found}/{total_req}")
        if total_req and found < int(MIN_PICK_COVERAGE * total_req):
            raise SystemExit(
                f"Bulk cache is missing {len(missing_paths)}/{total_req} selected "
                "parquets. Re-run with --bulk --force-download, or drop --bulk "
                f"to fetch just these files.\nExample missing: {missing_paths[:5]}"
            )
    else:
        _fetch_missing(api, picks)

    mode = "local zip" if local_zip else "full zip cache" if args.bulk else "targeted cache"
    print(
        f"Converting {len(picks)} clips for "
        f"{len({p.sign for p in picks})} signs "
        f"(workers={args.workers}, mode={mode}) ..."
    )
    _clear_prior_gislr_samples()
    counts: dict[str, int] = {}
    done = 0
    skipped = 0

    def _report(label_feats=None, exc=None) -> None:
        nonlocal done, skipped
        done += 1
        if exc is not None:
            skipped += 1
            print(f"  skipped one clip: {_format_skip(exc)}")
        else:
            _save_feats(counts, *label_feats)
        if done % 25 == 0 or done == len(picks):
            print(f"  {done}/{len(picks)} clips ({sum(counts.values())} samples)")

    # Everything is on disk by now, so conversion never hits the network.
    if args.workers == 1:
        for row in picks:
            try:
                _report(_process(row))
            except Exception as exc:  # noqa: BLE001
                _report(exc=exc)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_process, r) for r in picks]
            for fut in as_completed(futures):
                try:
                    _report(fut.result())
                except Exception as exc:  # noqa: BLE001
                    _report(exc=exc)

    total = sum(counts.values())
    print("\nImported samples per label:")
    if not counts:
        print("  (none)")
    else:
        for label, n in sorted(counts.items()):
            print(f"  {label:<10} {n}")
    print(f"Total samples: {total}  (skipped clips: {skipped})")
    print(f"Labels with samples: {len(counts)}")

    # Full mapped set is 24 signs; a few may be absent from GISLR. Expect ~20+.
    if total == 0 or len(counts) < 15:
        print(
            "\nERROR: import incomplete "
            f"(samples={total}, labels={len(counts)}).\n"
            "Re-run to retry failed downloads, or use --force-download to start "
            "from an empty cache.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("\nNext: record extra '_none' samples, then run: python train.py")


if __name__ == "__main__":
    main()
