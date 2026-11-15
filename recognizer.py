"""SignRecognizer: frames → sign tokens → finished sentences."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np

import config
from landmarks import HandTracker


@dataclass
class RecognitionEvent:
    """A finished sentence of sign tokens ready for translation."""

    tokens: List[str]
    finished_at: float = field(default_factory=time.time)


class SignRecognizer:
    """
    Feed frames; get emitted tokens and finished sentences.

    Smoothing rules (from spec):
    - Emit only when the same label has probability >= 0.75 for 6 consecutive frames
    - Never emit `_none`
    - Don't repeat the same sign until hands rest (`_none`) or a different sign appears
    - A sentence ends after 1.5 s with no new sign
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        tracker: Optional[HandTracker] = None,
    ) -> None:
        path = model_path or config.CLASSIFIER_PATH
        if not path.exists():
            raise SystemExit(
                f"Classifier not found at {path}. "
                "Collect samples with `python collect.py`, then run `python train.py`."
            )
        payload = joblib.load(path)
        self._pipeline = payload["pipeline"]
        self._encoder = payload["label_encoder"]
        self.tracker = tracker or HandTracker()
        self._owns_tracker = tracker is None

        self.current_label = "_none"
        self.current_confidence = 0.0
        self.tokens: List[str] = []
        self._streak_label: Optional[str] = None
        self._streak_count = 0
        # After rest we set this to "_none" so the same sign can be re-emitted.
        self._last_emitted: Optional[str] = None
        self._last_emit_time: Optional[float] = None
        self._pending_sentence = False

    def process_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[str], Optional[RecognitionEvent], object]:
        """
        Process one BGR frame.

        Returns:
            emitted_token: newly emitted sign or None
            finished_event: RecognitionEvent when a sentence ends, else None
            landmark_result: raw MediaPipe result (for drawing)
        """
        _, result = self.tracker.process(frame_bgr)
        feats = self.tracker.features()
        emitted: Optional[str] = None
        finished: Optional[RecognitionEvent] = None

        hands_now = bool(getattr(result, "hand_landmarks", None))
        if feats is not None and not feats.any():
            # No hands for the whole window: that is rest, not a sign. The
            # model never saw all-zero input and would guess (e.g. WHERE).
            self.current_label = "_none"
            self.current_confidence = 1.0
            self._streak_label = None
            self._streak_count = 0
            self._last_emitted = "_none"
        elif feats is not None and not hands_now:
            # Hand dropped out this frame: hold off rather than classify a
            # window that ends in zeros.
            self._streak_label = None
            self._streak_count = 0
        elif feats is not None:
            proba = self._pipeline.predict_proba([feats])[0]
            best_i = int(np.argmax(proba))
            label = str(self._encoder.classes_[best_i])
            conf = float(proba[best_i])
            self.current_label = label
            self.current_confidence = conf

            if conf >= config.CONFIDENCE_THRESHOLD:
                if label == self._streak_label:
                    self._streak_count += 1
                else:
                    self._streak_label = label
                    self._streak_count = 1
            else:
                self._streak_label = None
                self._streak_count = 0

            if label == "_none" and conf >= config.CONFIDENCE_THRESHOLD:
                # Hands resting — allow the next (even same) sign to fire.
                self._last_emitted = "_none"

            elif (
                self._streak_count >= config.CONSECUTIVE_FRAMES
                and label != "_none"
                and label != self._last_emitted
            ):
                emitted = label
                self.tokens.append(label)
                self._last_emitted = label
                self._last_emit_time = time.time()
                self._pending_sentence = True
                self._streak_count = 0

        if (
            self._pending_sentence
            and self._last_emit_time is not None
            and (time.time() - self._last_emit_time) >= config.SENTENCE_GAP_SEC
            and self.tokens
        ):
            finished = RecognitionEvent(tokens=list(self.tokens))
            self.tokens.clear()
            self._pending_sentence = False
            self._last_emit_time = None

        return emitted, finished, result

    def clear_tokens(self) -> None:
        self.tokens.clear()
        self._pending_sentence = False
        self._last_emit_time = None

    def close(self) -> None:
        if self._owns_tracker:
            self.tracker.close()
