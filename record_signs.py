"""Record your own signs for words the dataset does not have.

The Kaggle dataset has no I, YOU, HELP, MEDICINE or PHARMACY. Record them
here, then run `python train.py` to teach them to the model.

  python record_signs.py I HELP MEDICINE PHARMACY YOU
  python record_signs.py HELP --count 20

For each word: raise your hands, make the sign once, lower your hands.
Each time you do that one recording is saved. Vary it a little (speed,
position) so the model learns the sign rather than one exact movement.

Keys: N = next word, D = delete the last recording, Q = quit.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import config
from landmarks import BodyHandTracker, draw_landmarks
from recognizer import SignSegmenter
from sources import WebcamSource

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("words", nargs="+", help="words to record, e.g. I HELP MEDICINE")
    ap.add_argument("--count", type=int, default=15, help="recordings per word")
    args = ap.parse_args()
    words = [w.upper() for w in args.words]

    source = WebcamSource()
    source.open()
    tracker = BodyHandTracker()
    segmenter = SignSegmenter()
    t0 = time.time()
    wi = 0
    saved: list = []
    try:
        while wi < len(words):
            word = words[wi]
            out_dir = config.CUSTOM_SIGNS_DIR / word
            out_dir.mkdir(parents=True, exist_ok=True)
            have = len(list(out_dir.glob("*.npz")))

            ok, frame = source.read()
            if not ok or frame is None:
                break
            now = time.time()
            h, w = frame.shape[:2]
            left, right, pose = tracker.process(frame, timestamp_ms=int((now - t0) * 1000))
            segment = segmenter.feed(left, right, pose, w / h, now)
            if segment is not None:
                path = out_dir / f"{int(now * 1000)}.npz"
                np.savez_compressed(
                    path, lh=segment[0], rh=segment[1], pose=segment[2], aspect=w / h
                )
                saved.append(path)
                have += 1
                print(f"  saved {word} #{have}")
                if have >= args.count:
                    wi += 1
                    continue

            draw_landmarks(frame, tracker.last_hand_result)
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)
            state = "Recording... lower your hands when done" if segmenter.signing else "Raise your hands and sign"
            cv2.putText(frame, f"{word}   {have}/{args.count}", (10, 28), _FONT, 0.8, (120, 230, 120), 2)
            cv2.putText(frame, state, (10, 58), _FONT, 0.55, (0, 190, 255) if segmenter.signing else (200, 200, 200), 1)
            cv2.putText(frame, "N=next word  D=delete last  Q=quit", (10, h - 10), _FONT, 0.45, (200, 200, 200), 1)
            cv2.imshow("Hermes - record signs", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("n"), ord("N")):
                wi += 1
            if key in (ord("d"), ord("D")) and saved:
                last = saved.pop()
                last.unlink(missing_ok=True)
                print(f"  deleted {last.parent.name}/{last.name}")
    finally:
        tracker.close()
        source.close()
        cv2.destroyAllWindows()
    print("Done. Now run: python train.py")


if __name__ == "__main__":
    main()
