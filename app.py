"""Hermes in-person mode: camera + overlay + voice + captions."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from typing import List, Optional, Tuple

import cv2

import config
from landmarks import draw_landmarks
from recognizer import SignRecognizer
from sources import VideoFileSource, WebcamSource
from translate import translate
from voice import speak, transcribe

# Set by SIGINT/SIGTERM so the loop can exit without a raw "terminated" death.
_stop = False


def _request_stop(signum, _frame) -> None:
    global _stop
    _stop = True
    name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    print(f"\n[app] received {name} — quitting…")


def _window_closed(name: str = "Hermes") -> bool:
    """True if the HighGUI window was closed (e.g. macOS red traffic-light)."""
    try:
        # -1 means the window does not exist / was destroyed.
        prop = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
        return prop < 1
    except Exception:  # noqa: BLE001
        return True


def _safe_destroy_windows() -> None:
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:  # noqa: BLE001
        pass


def _draw_panel(
    frame,
    label: str,
    confidence: float,
    tokens: List[str],
    signer_caption: str,
    reply_caption: str,
) -> None:
    h, w = frame.shape[:2]
    # Top status
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)
    pred = f"{label}  ({confidence:.0%})"
    cv2.putText(frame, pred, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 180), 2)
    token_line = "Tokens: " + (" ".join(tokens) if tokens else "—")
    cv2.putText(frame, token_line[:70], (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # Bottom caption panel
    panel_h = 90
    cv2.rectangle(frame, (0, h - panel_h), (w, h), (15, 15, 15), -1)
    cv2.putText(
        frame,
        f"Signer: {signer_caption or '—'}"[:80],
        (10, h - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"Reply: {reply_caption or '—'}"[:80],
        (10, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 220, 255),
        1,
    )
    cv2.putText(
        frame,
        "R=record reply  C=clear  Q=quit",
        (w - 310, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (140, 140, 140),
        1,
    )


class _Conversation:
    """Captions + dialogue history, updated from background threads.

    Translation, speech and reply recording each take a second or more, so
    they run off the video loop; the loop just reads the latest captions.
    """

    def __init__(self, voice_enabled: bool) -> None:
        self.voice_enabled = voice_enabled
        self.lock = threading.Lock()
        self.history: List[Tuple[str, str]] = []
        self.signer = ""
        self.reply = ""
        self.listening = False
        self._generation = 0  # bumped on clear so stale results are dropped

    def captions(self) -> Tuple[str, str]:
        with self.lock:
            return self.signer, self.reply

    def clear(self) -> None:
        with self.lock:
            self.history.clear()
            self.signer = ""
            self.reply = "Listening..." if self.listening else ""
            self._generation += 1

    def on_sentence(self, tokens: List[str]) -> None:
        with self.lock:
            gen = self._generation
            history = list(self.history)
            self.signer = "Translating: " + " ".join(tokens)
        threading.Thread(
            target=self._translate, args=(tokens, history, gen), daemon=True
        ).start()

    def _translate(self, tokens: List[str], history, gen: int) -> None:
        english = translate(tokens, conversation=history)
        with self.lock:
            if gen != self._generation:
                return
            self.signer = english
            self.history.append(("signer", english))
        print(f"Signer: {english}")
        if self.voice_enabled and english:
            try:
                speak(english)
            except Exception as exc:  # noqa: BLE001
                print(f"[app] speak failed: {exc}")

    def record_reply(self) -> None:
        with self.lock:
            if self.listening:
                return
            self.listening = True
            self.reply = f"Listening... ({config.RECORD_SECONDS}s)"
            gen = self._generation
        print("[app] recording reply…")
        threading.Thread(target=self._record, args=(gen,), daemon=True).start()

    def _record(self, gen: int) -> None:
        try:
            text = transcribe(seconds=config.RECORD_SECONDS)
        except Exception as exc:  # noqa: BLE001
            print(f"[app] transcription failed: {exc}")
            text = ""
        with self.lock:
            self.listening = False
            if gen != self._generation:
                self.reply = ""
                return
            self.reply = text or "(no speech heard)"
            if text:
                self.history.append(("reply", text))
        print(f"Reply: {text}")


def run(video: Optional[str] = None, no_voice: bool = False) -> None:
    global _stop
    _stop = False
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    # Fail fast if the classifier is missing — before grabbing the camera.
    recognizer = SignRecognizer()

    if video:
        source = VideoFileSource(video, loop=True)
    else:
        source = WebcamSource()
    source.open()

    convo = _Conversation(voice_enabled=not no_voice)

    print("Hermes in-person mode")
    print("Q quit | R record reply (5s) | C clear captions")
    print("(Prefer Q over the window red button on macOS.)")
    if no_voice:
        print("Voice disabled (--no-voice)")

    try:
        while not _stop:
            ok, frame = source.read()
            if not ok or frame is None:
                print("No more frames.")
                break

            _, finished, result = recognizer.process_frame(frame)
            draw_landmarks(frame, result)

            if finished is not None:
                convo.on_sentence(finished.tokens)

            signer_caption, reply_caption = convo.captions()
            _draw_panel(
                frame,
                recognizer.current_label,
                recognizer.current_confidence,
                recognizer.tokens,
                signer_caption,
                reply_caption,
            )
            cv2.imshow("Hermes", frame)

            key = cv2.waitKey(1) & 0xFF
            if _window_closed("Hermes"):
                print("[app] window closed — quitting…")
                break
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                convo.clear()
                recognizer.clear_tokens()
                print("[app] captions cleared")
            if key in (ord("r"), ord("R")):
                convo.record_reply()
    finally:
        try:
            recognizer.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            source.close()
        except Exception:  # noqa: BLE001
            pass
        _safe_destroy_windows()
        print("[app] stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes in-person sign translator")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Play a recorded video clip instead of the webcam (backup demo)",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Skip ElevenLabs TTS playback",
    )
    args = parser.parse_args()
    try:
        run(video=args.video, no_voice=args.no_voice)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[app] fatal: {exc}", file=sys.stderr)
        _safe_destroy_windows()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
