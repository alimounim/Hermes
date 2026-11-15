"""Record training samples from the webcam.

Controls:
  SPACE — capture one sample of the current sign
  N     — advance to the next sign in the vocabulary
  Q     — quit
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

import config
from landmarks import HandTracker, draw_landmarks, window_features
from sources import WebcamSource


def _sign_dir(label: str) -> Path:
    d = config.SAMPLES_DIR / label
    d.mkdir(parents=True, exist_ok=True)
    return d


def _count_samples(label: str) -> int:
    d = _sign_dir(label)
    return len(list(d.glob("*.npy")))


def main() -> None:
    labels = list(config.VOCABULARY)
    idx = 0
    print("Hermes data collection")
    print("SPACE = record sample | N = next sign | Q = quit")
    print(f"Target: {config.SAMPLES_PER_SIGN} samples per sign\n")

    tracker = HandTracker()
    source = WebcamSource()
    source.open()

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                print("Camera frame read failed.")
                break

            vec, result = tracker.process(frame)
            draw_landmarks(frame, result)

            label = labels[idx]
            count = _count_samples(label)
            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"Sign: {label}  ({count}/{config.SAMPLES_PER_SIGN})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                overlay,
                "SPACE=record  N=next  Q=quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            hands_ok = bool(result.hand_landmarks) or label == "_none"
            status = "hands OK" if hands_ok else "show hands"
            cv2.putText(
                overlay,
                status,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if hands_ok else (0, 0, 255),
                2,
            )
            cv2.imshow("Hermes Collect", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("n"), ord("N")):
                idx = (idx + 1) % len(labels)
                tracker.reset_window()
                print(f"→ Now recording: {labels[idx]}")
                continue
            if key == ord(" "):
                # Need a full window of frames for moving signs.
                # Fill the rest of the window quickly from live feed.
                window = list(tracker._window)
                while len(window) < config.WINDOW_SIZE:
                    ok2, frame2 = source.read()
                    if not ok2 or frame2 is None:
                        break
                    v2, _ = tracker.process(frame2)
                    window = list(tracker._window)
                    time.sleep(0.01)

                if len(window) < config.WINDOW_SIZE:
                    # Fall back: pad with the latest vector
                    while len(window) < config.WINDOW_SIZE:
                        window.append(vec.copy())

                feats = window_features(window)
                out_dir = _sign_dir(label)
                stamp = int(time.time() * 1000)
                out_path = out_dir / f"{stamp}.npy"
                np.save(out_path, feats)
                print(f"Saved {out_path.name} for {label} "
                      f"({_count_samples(label)}/{config.SAMPLES_PER_SIGN})")
                tracker.reset_window()
    finally:
        tracker.close()
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
