"""Frame sources: live webcam and recorded video file."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

import config


class WebcamSource:
    """Capture frames from a local camera."""

    def __init__(self, index: int = config.CAMERA_INDEX) -> None:
        self.index = index
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise SystemExit(
                f"Could not open camera index {self.index}. "
                "Check that a webcam is connected and not used by another app."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("WebcamSource is not open. Call open() first.")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def frames(self) -> Iterator[np.ndarray]:
        self.open()
        try:
            while True:
                ok, frame = self.read()
                if not ok or frame is None:
                    break
                yield frame
        finally:
            self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "WebcamSource":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()


class VideoFileSource:
    """Play frames from a recorded video (backup demo)."""

    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = path
        self.loop = loop
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise SystemExit(f"Could not open video file: {self.path}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("VideoFileSource is not open. Call open() first.")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    return False, None
                return True, frame
            return False, None
        return True, frame

    def frames(self) -> Iterator[np.ndarray]:
        self.open()
        try:
            while True:
                ok, frame = self.read()
                if not ok or frame is None:
                    break
                yield frame
        finally:
            self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoFileSource":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()
