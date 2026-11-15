"""SignRecognizer: camera frames → whole signs → finished sentences.

How it decides what is one sign:

1. Hands come up (above the "rest line", about elbow height)  → a sign starts.
2. Hands go back down or leave the picture for SIGN_END_SEC    → the sign ends.
3. The whole movement is classified once. Confident → added to the sentence;
   unsure or unknown → shown as "not sure" and NOT added.
4. No new sign for SENTENCE_GAP_SEC → the sentence is finished.

Judging the whole movement (instead of guessing on every frame) is what keeps
half-finished movements between signs from turning into words.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np

import config
import signfeat
from landmarks import BodyHandTracker

MODEL_KIND = "segment-v1"


@dataclass
class RecognitionEvent:
    """A finished sentence of sign tokens ready for translation."""

    tokens: List[str]
    finished_at: float = field(default_factory=time.time)


@dataclass
class SignResult:
    """Outcome of judging one sign."""

    accepted: bool
    guesses: List[Tuple[str, float]]  # best first, at most 3
    at: float


def load_classifier(path: Optional[Path] = None):
    path = path or config.CLASSIFIER_PATH
    if not path.exists():
        raise SystemExit(f"Classifier not found at {path}. Run `python train.py` first.")
    payload = joblib.load(path)
    if payload.get("kind") != MODEL_KIND:
        raise SystemExit(
            f"{path} is an old-style model. Rebuild it:\n"
            "  python build_dataset.py   # once, needs data/asl-signs/\n"
            "  python train.py"
        )
    return payload["pipeline"]


class SignSegmenter:
    """Cuts the landmark stream into signs: hands up → ... → hands down."""

    def __init__(self) -> None:
        self.signing = False
        self.body: Optional[Tuple[np.ndarray, float]] = None  # (nose, shoulder width)
        self._frames: list = []
        self._hand_frames = 0
        self._sign_started = 0.0
        self._last_hands_up = 0.0

    def feed(self, left, right, pose, aspect: float, now: float):
        """Add one frame. Returns (lh, rh, pose) arrays when a sign just ended."""
        seen = bool(np.isfinite(left[0, 0]) or np.isfinite(right[0, 0]))
        left, right = self._drop_resting_hands(left, right, pose, aspect)
        hands_up = bool(np.isfinite(left[0, 0]) or np.isfinite(right[0, 0]))

        if hands_up:
            if not self.signing:
                self.signing = True
                self._frames = []
                self._hand_frames = 0
                self._sign_started = now
            self._last_hands_up = now
            self._hand_frames += 1
        if not self.signing:
            return None

        self._frames.append((left, right, pose))
        # Hands visibly resting end the sign quickly. Hands that simply vanished
        # may only be motion blur mid-sign, so wait longer before judging.
        patience = config.SIGN_END_SEC if seen else config.SIGN_LOST_SEC
        ended = not hands_up and now - self._last_hands_up >= patience
        too_long = now - self._sign_started >= config.SIGN_MAX_SEC
        if not (ended or too_long):
            return None
        frames, n_hand = self._frames, self._hand_frames
        self.signing = False
        self._frames = []
        if n_hand < config.SIGN_MIN_FRAMES:
            return None  # a flicker, not a sign
        return tuple(np.stack([f[i] for f in frames]) for i in range(3))

    def _drop_resting_hands(self, left, right, pose, aspect):
        """Hands below the rest line count as not signing."""
        nose, lsh, rsh = pose[0], pose[1], pose[2]
        if np.isfinite(pose[:3]).all():
            sw = float(np.hypot((lsh[0] - rsh[0]) * aspect, lsh[1] - rsh[1]))
            if sw > 1e-4:
                self.body = (nose, sw)
        if self.body is None:
            return left, right
        nose, sw = self.body
        out = []
        for hand in (left, right):
            resting = (
                np.isfinite(hand[0, 1])
                and (hand[0, 1] - nose[1]) / sw > config.REST_BELOW_NOSE
            )
            out.append(np.full_like(hand, np.nan) if resting else hand)
        return out[0], out[1]


class SignRecognizer:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        tracker: Optional[BodyHandTracker] = None,
    ) -> None:
        self._pipeline = load_classifier(model_path)
        self._classes = [str(c) for c in self._pipeline.classes_]
        self.tracker = tracker or BodyHandTracker()
        self._owns_tracker = tracker is None
        self._segmenter = SignSegmenter()

        self.tokens: List[str] = []
        self.last_result: Optional[SignResult] = None
        self._last_emit_time: Optional[float] = None
        self._t0 = time.time()

    @property
    def signing(self) -> bool:
        return self._segmenter.signing

    @property
    def body(self):
        return self._segmenter.body

    # -- per frame ---------------------------------------------------------

    def process_frame(
        self, frame_bgr: np.ndarray, now: Optional[float] = None
    ) -> Tuple[Optional[str], Optional[RecognitionEvent], object]:
        """Returns (sign added this frame, finished sentence, raw hand result)."""
        now = time.time() if now is None else now
        h, w = frame_bgr.shape[:2]
        left, right, pose = self.tracker.process(
            frame_bgr, timestamp_ms=int((now - self._t0) * 1000)
        )
        emitted = self.feed(left, right, pose, w / h, now)
        finished = self._maybe_finish_sentence(now)
        return emitted, finished, getattr(self.tracker, "last_hand_result", None)

    def feed(self, left, right, pose, aspect: float, now: float) -> Optional[str]:
        """Advance with one frame of landmarks. Split out so it can be tested
        with dataset clips instead of a camera."""
        segment = self._segmenter.feed(left, right, pose, aspect, now)
        if segment is None:
            return None
        return self._judge(segment, aspect, now)

    # -- judging one sign ----------------------------------------------------

    def _judge(self, segment, aspect: float, now: float) -> Optional[str]:
        guesses = self.classify(*segment, aspect)
        if not guesses:
            return None
        label, conf = guesses[0]
        accepted = label != "_other" and conf >= config.SIGN_CONFIDENCE
        self.last_result = SignResult(accepted=accepted, guesses=guesses, at=now)
        if not accepted:
            return None
        self.tokens.append(label)
        self._last_emit_time = now
        return label

    def classify(self, lh, rh, pose, aspect: float) -> List[Tuple[str, float]]:
        """Top-3 (sign, probability) for one whole movement."""
        feats = signfeat.segment_features(lh, rh, pose, aspect)
        if feats is None:
            return []
        proba = self._pipeline.predict_proba([feats])[0]
        order = np.argsort(-proba)[:3]
        return [(self._classes[i], float(proba[i])) for i in order]

    # -- sentences -----------------------------------------------------------

    def _maybe_finish_sentence(self, now: float) -> Optional[RecognitionEvent]:
        if (
            self.tokens
            and not self.signing
            and self._last_emit_time is not None
            and now - self._last_emit_time >= config.SENTENCE_GAP_SEC
        ):
            event = RecognitionEvent(tokens=list(self.tokens))
            self.clear_tokens()
            return event
        return None

    def sentence_seconds_left(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds until the sentence is sent, or None (nothing pending / mid-sign)."""
        if not self.tokens or self.signing or self._last_emit_time is None:
            return None
        now = time.time() if now is None else now
        return max(0.0, config.SENTENCE_GAP_SEC - (now - self._last_emit_time))

    def undo_last(self) -> Optional[str]:
        """Remove the most recent sign from the sentence being built."""
        if not self.tokens:
            return None
        self._last_emit_time = time.time()
        return self.tokens.pop()

    def clear_tokens(self) -> None:
        self.tokens.clear()
        self._last_emit_time = None

    def close(self) -> None:
        if self._owns_tracker:
            self.tracker.close()
