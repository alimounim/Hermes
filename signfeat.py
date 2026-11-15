"""Whole-sign features shared by training (GISLR clips) and the live recognizer.

One sign = one segment: the frames from hands coming up to hands going down.
The segment becomes a fixed-size vector that captures

- hand SHAPE   (each hand, wrist-centred and size-normalised),
- hand PLACE   (wrist and index fingertip relative to the nose, in
                shoulder-widths — this is what separates HELLO at the forehead
                from BED at the cheek),
- MOVEMENT     (the above sampled at T evenly spaced moments).

Left-handed signing is mirrored so the busier hand is always in slot 0.
All inputs are image-normalised MediaPipe coordinates; `aspect` (frame
width / height) makes x and y comparable before anything is measured.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

T_STEPS = 16
MIN_FRAMES = 4  # frames with a visible hand needed to call it a sign
# Aspect (w/h) of the GISLR recordings: 3:4 portrait, verified from face proportions.
GISLR_ASPECT = 0.75

_TIPS = [4, 8, 12, 16, 20]
_TIP_PAIRS = [(a, b) for i, a in enumerate(_TIPS) for b in _TIPS[i + 1:]]
# shape, fingertip spreads, wrist place, index-tip place, size, present
_HAND_DIM = 63 + len(_TIP_PAIRS) + len(_TIPS) + 2 + 2 + 1 + 1
FRAME_DIM = 2 * _HAND_DIM + 2  # + offset between the two wrists
# T snapshots of the movement + its average and spread over time
FEATURE_DIM = (T_STEPS + 2) * FRAME_DIM

# Indices into the kept pose landmarks (config.POSE_KEEP): nose, shoulders.
_NOSE, _LSH, _RSH = 0, 1, 2


def _hand_block(hand: np.ndarray, present: np.ndarray, nose: np.ndarray, sw: float):
    """Per-frame features for one hand. hand: (n, 21, 3) with NaN when absent."""
    h = np.nan_to_num(hand, nan=0.0)
    rel = h - h[:, :1]
    size = np.linalg.norm(rel[:, :, :2], axis=2).max(axis=1)
    safe = np.where(size > 1e-6, size, 1.0)
    unit = rel / safe[:, None, None]
    shape = unit.reshape(len(h), -1)
    # Distances between fingertips (and to the wrist) do not change when the
    # hand is rotated, so they carry handshape across different signers.
    pa, pb = [a for a, _ in _TIP_PAIRS], [b for _, b in _TIP_PAIRS]
    spread = np.linalg.norm(unit[:, pa] - unit[:, pb], axis=2)
    reach = np.linalg.norm(unit[:, _TIPS], axis=2)
    place = (h[:, 0, :2] - nose) / sw
    tip = (h[:, 8, :2] - nose) / sw
    block = np.concatenate(
        [shape, spread, reach, place, tip, (size / sw)[:, None], np.ones((len(h), 1), np.float32)],
        axis=1,
    )
    block[~present] = 0.0
    return block, place


def _add_travel(dom, oth, dp, op, sw, rng):
    """Augmentation: add the hands travelling up into / down out of the sign.

    Dataset clips mostly start with the hand already in place; a live segment
    also contains the trip from the lap and back. Fake that trip by sliding the
    first / last frame in from below.
    """
    n = len(dom)
    for at_start in (True, False):
        if rng.random() < 0.25:
            continue
        k = int(rng.integers(1, max(2, n // 4) + 1))
        ramp = np.linspace(1.0, 0.0, k + 1, dtype=np.float32)[:-1]
        if not at_start:
            ramp = ramp[::-1]
        shift = np.zeros((k, 1, 3), np.float32)
        shift[:, 0, 1] = ramp * rng.uniform(0.3, 1.0) * sw
        shift[:, 0, 0] = ramp * rng.normal(0, 0.15) * sw
        i = 0 if at_start else n - 1
        d_new, o_new = dom[i][None] + shift, oth[i][None] + shift
        dp_new, op_new = np.repeat(dp[i], k), np.repeat(op[i], k)
        order = (lambda new, old: np.concatenate([new, old])) if at_start else (
            lambda new, old: np.concatenate([old, new])
        )
        dom, oth = order(d_new, dom), order(o_new, oth)
        dp, op = order(dp_new, dp), order(op_new, op)
    return dom, oth, dp, op


def segment_features(
    lh: np.ndarray,
    rh: np.ndarray,
    pose: np.ndarray,
    aspect: float,
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """Turn one sign segment into a FEATURE_DIM vector, or None if too short.

    lh, rh: (n, 21, 3) left/right hand landmarks, NaN rows where not detected.
    pose:   (n, 7, 2) nose, shoulders, elbows, wrists (NaN where not detected).
    rng:    if given, apply training-time augmentation.
    """
    lh = np.array(lh, dtype=np.float32)
    rh = np.array(rh, dtype=np.float32)
    pose = np.array(pose, dtype=np.float32)
    lp = ~np.isnan(lh[:, 0, 0])
    rp = ~np.isnan(rh[:, 0, 0])
    keep = lp | rp
    if keep.sum() < MIN_FRAMES:
        return None
    lh, rh, pose, lp, rp = lh[keep], rh[keep], pose[keep], lp[keep], rp[keep]

    if rng is not None:
        # Trim the ends: live segments never start/stop exactly like a dataset clip.
        n = len(lh)
        a = rng.integers(0, max(1, int(n * 0.2)) + 1)
        b = n - rng.integers(0, max(1, int(n * 0.2)) + 1)
        if b - a >= MIN_FRAMES:
            lh, rh, pose, lp, rp = lh[a:b], rh[a:b], pose[a:b], lp[a:b], rp[a:b]
        aspect = aspect * rng.uniform(0.88, 1.14)

    # Make x and y comparable (z is in x units).
    for arr in (lh, rh):
        arr[:, :, 0] *= aspect
        arr[:, :, 2] *= aspect
    pose[:, :, 0] *= aspect

    # Busier hand goes first; mirror so it is always a "right" hand.
    if lp.sum() > rp.sum():
        lh, rh, lp, rp = rh, lh, rp, lp
        for arr in (lh, rh):
            arr[:, :, 0] *= -1
        pose[:, :, 0] *= -1
    dom, oth, dp, op = rh, lh, rp, lp

    # Body frame: nose as origin, shoulder width as the unit.
    with np.errstate(all="ignore"):
        nose = np.nanmedian(pose[:, _NOSE], axis=0)
        sw = float(np.nanmedian(np.linalg.norm(pose[:, _LSH] - pose[:, _RSH], axis=1)))
    if not np.isfinite(nose).all() or not np.isfinite(sw) or sw < 1e-4:
        return None

    if rng is not None:
        ang = rng.uniform(-0.17, 0.17)  # ~10 degrees of camera/arm tilt
        c, s = np.cos(ang), np.sin(ang)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        for arr in (dom, oth):
            arr[:, :, :2] = (arr[:, :, :2] - nose) @ rot.T + nose
        sw *= rng.uniform(0.9, 1.1)
        nose = nose + rng.normal(0, 0.06, 2).astype(np.float32) * sw
        if rng.random() < 0.6:
            dom, oth, dp, op = _add_travel(dom, oth, dp, op, sw, rng)

    d_block, d_place = _hand_block(dom, dp, nose, sw)
    o_block, o_place = _hand_block(oth, op, nose, sw)
    between = np.where((dp & op)[:, None], o_place - d_place, 0.0)
    frames = np.concatenate([d_block, o_block, between], axis=1).astype(np.float32)

    idx = np.round(np.linspace(0, len(frames) - 1, T_STEPS)).astype(int)
    return np.concatenate([frames[idx].reshape(-1), frames.mean(axis=0), frames.std(axis=0)])
