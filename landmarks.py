"""MediaPipe Tasks HandLandmarker + landmark feature helpers."""

from __future__ import annotations

import urllib.request
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config


def ensure_hand_model(path: Path = config.HAND_LANDMARKER_MODEL) -> Path:
    """Download the HandLandmarker .task model into models/ if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"Downloading HandLandmarker model to {path} ...")
    try:
        urllib.request.urlretrieve(config.HAND_LANDMARKER_URL, path)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Failed to download HandLandmarker model from "
            f"{config.HAND_LANDMARKER_URL}\n{exc}"
        ) from exc
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"HandLandmarker model missing or empty at {path}")
    print("Download complete.")
    return path


def _normalize_hand(points: np.ndarray) -> np.ndarray:
    """Subtract wrist, scale by max 2D distance from wrist. Shape (21, 3)."""
    wrist = points[0].copy()
    points = points - wrist
    dists = np.linalg.norm(points[:, :2], axis=1)
    scale = float(dists.max())
    if scale > 1e-6:
        points = points / scale
    return points


def landmarks_to_vector(result: mp_vision.HandLandmarkerResult) -> np.ndarray:
    """
    Pack detected hands into a 126-dim vector.
    Slot by handedness (Left / Right); zeros if a hand is missing.
    Each hand is wrist-normalized and size-normalized.
    """
    left = np.zeros((config.LANDMARKS_PER_HAND, config.COORDS_PER_LANDMARK), dtype=np.float32)
    right = np.zeros((config.LANDMARKS_PER_HAND, config.COORDS_PER_LANDMARK), dtype=np.float32)

    if not result.hand_landmarks:
        return np.concatenate([left.ravel(), right.ravel()])

    for i, hand_lms in enumerate(result.hand_landmarks):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
        pts = _normalize_hand(pts)
        label = "Unknown"
        if result.handedness and i < len(result.handedness):
            cats = result.handedness[i]
            if cats:
                label = cats[0].category_name
        if label.lower().startswith("left"):
            left = pts
        else:
            # Default unknown / Right into the right slot
            right = pts

    return np.concatenate([left.ravel(), right.ravel()]).astype(np.float32)


def window_features(window: Deque[np.ndarray] | list[np.ndarray]) -> np.ndarray:
    """
    Sliding-window features from up to WINDOW_SIZE frames:
    concat(last, mean, last - first, std).
    """
    arr = np.stack(list(window), axis=0).astype(np.float32)
    last = arr[-1]
    mean = arr.mean(axis=0)
    delta = arr[-1] - arr[0]
    std = arr.std(axis=0) if len(arr) > 1 else np.zeros_like(last)
    return np.concatenate([last, mean, delta, std]).astype(np.float32)


class HandTracker:
    """Wraps MediaPipe Tasks HandLandmarker in VIDEO mode."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        model_path = ensure_hand_model(model_path or config.HAND_LANDMARKER_MODEL)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self._window: Deque[np.ndarray] = deque(maxlen=config.WINDOW_SIZE)
        self._last_result: Optional[mp_vision.HandLandmarkerResult] = None

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, mp_vision.HandLandmarkerResult]:
        """Run detection on a BGR OpenCV frame. Returns (126-vector, raw result)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33  # ~30 FPS
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        vec = landmarks_to_vector(result)
        self._window.append(vec)
        self._last_result = result
        return vec, result

    def features(self) -> Optional[np.ndarray]:
        """Return window features once the buffer is full; else None."""
        if len(self._window) < config.WINDOW_SIZE:
            return None
        return window_features(self._window)

    def reset_window(self) -> None:
        self._window.clear()

    @property
    def last_result(self) -> Optional[mp_vision.HandLandmarkerResult]:
        return self._last_result

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# MediaPipe hand topology (start, end) landmark indices.
_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def draw_landmarks(frame_bgr: np.ndarray, result: Optional[mp_vision.HandLandmarkerResult]) -> None:
    """Draw hand landmarks + connections onto a BGR frame in-place."""
    if result is None or not result.hand_landmarks:
        return
    h, w = frame_bgr.shape[:2]
    for hand_lms in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for start, end in _HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts[start], pts[end], (0, 200, 120), 2)
        for p in pts:
            cv2.circle(frame_bgr, p, 3, (0, 255, 255), -1)
