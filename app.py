"""Hermes in-person mode: camera + overlay + voice + captions."""

from __future__ import annotations

import argparse
import threading
from typing import List, Optional, Tuple

import cv2

import config
from landmarks import draw_landmarks
from recognizer import SignRecognizer
from sources import VideoFileSource, WebcamSource
from translate import translate
from voice import speak, transcribe


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


def _speak_async(text: str, enabled: bool) -> None:
    if not enabled or not text:
        return

    def _run() -> None:
        try:
            speak(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[app] speak failed: {exc}")

    threading.Thread(target=_run, daemon=True).start()


def run(video: Optional[str] = None, no_voice: bool = False) -> None:
    if video:
        source = VideoFileSource(video, loop=True)
    else:
        source = WebcamSource()
    source.open()

    recognizer = SignRecognizer()
    conversation: List[Tuple[str, str]] = []
    signer_caption = ""
    reply_caption = ""
    voice_enabled = not no_voice

    print("Hermes in-person mode")
    print("Q quit | R record reply (5s) | C clear captions")
    if no_voice:
        print("Voice disabled (--no-voice)")

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                print("No more frames.")
                break

            _, finished, result = recognizer.process_frame(frame)
            draw_landmarks(frame, result)

            if finished is not None:
                english = translate(finished.tokens, conversation=conversation)
                signer_caption = english
                conversation.append(("signer", english))
                print(f"Signer: {english}")
                _speak_async(english, voice_enabled)

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
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                signer_caption = ""
                reply_caption = ""
                conversation.clear()
                recognizer.clear_tokens()
                print("[app] captions cleared")
            if key in (ord("r"), ord("R")):
                try:
                    print("[app] recording reply…")
                    # Pause briefly so the OpenCV window stays responsive messaging
                    reply = transcribe(seconds=config.RECORD_SECONDS)
                    reply_caption = reply
                    if reply:
                        conversation.append(("reply", reply))
                    print(f"Reply: {reply}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[app] transcription failed: {exc}")
    finally:
        recognizer.close()
        source.close()
        cv2.destroyAllWindows()


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
    run(video=args.video, no_voice=args.no_voice)


if __name__ == "__main__":
    main()
