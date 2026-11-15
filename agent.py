"""Hermes call mode: LiveKit agent for Chert FaceTime.

Subscribes to Chert bridge tracks (`facetime-remote-video` /
`facetime-remote-audio`), runs the Phase 1 sign recognizer on caller video,
speaks translated English into the call via ElevenLabs TTS, and logs final
speech captions from the caller.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv

# Load .env then .env.local (local overrides). Never print secrets.
_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)

# LiveKit ElevenLabs plugin reads ELEVEN_API_KEY; Hermes uses ELEVENLABS_API_KEY.
if os.getenv("ELEVENLABS_API_KEY") and not os.getenv("ELEVEN_API_KEY"):
    os.environ["ELEVEN_API_KEY"] = os.environ["ELEVENLABS_API_KEY"]

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    UserInputTranscribedEvent,
    WorkerOptions,
    cli,
)
from livekit.plugins import elevenlabs, silero

import config
from recognizer import SignRecognizer
from translate import translate

logger = logging.getLogger("hermes.agent")
logging.basicConfig(level=logging.INFO)

AGENT_NAME = os.getenv("AGENT_NAME", "hermes-agent")
FACETIME_VIDEO = "facetime-remote-video"
FACETIME_AUDIO = "facetime-remote-audio"
GREETING = (
    "Hi, I'm Hermes, your sign language interpreter. "
    "Sign toward the camera and I'll speak for you."
)


def _require_runtime() -> None:
    missing = []
    for key in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if not os.getenv(key):
            missing.append(key)
    if missing:
        raise SystemExit(
            "Missing LiveKit env: "
            + ", ".join(missing)
            + ". Put LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET "
            "(same project) in .env or .env.local — never paste them into chat. "
            "Create a project at https://cloud.livekit.io. "
            "Chert-managed LiveKit (`chert livekit`) is Shared-plan only and "
            "is not provisioned on this trial project yet."
        )
    if not config.ELEVENLABS_API_KEY:
        raise SystemExit(
            "ELEVENLABS_API_KEY is missing. Add it to .env / .env.local "
            "(see .env.example)."
        )
    if not config.CLASSIFIER_PATH.exists():
        raise SystemExit(
            f"Classifier not found at {config.CLASSIFIER_PATH}. "
            "Train with `python train.py` after samples exist "
            "(or wait for GISLR import), or run "
            "`python scripts/make_smoke_classifier.py` for a temporary smoke model."
        )


def _bridge_ready(metadata: Optional[str]) -> bool:
    try:
        return json.loads(metadata or "{}").get("chert_bridge_ready") is True
    except Exception:  # noqa: BLE001
        return False


def _video_frame_to_bgr(frame: rtc.VideoFrame) -> Optional[np.ndarray]:
    """Convert a LiveKit video frame to BGR for OpenCV / MediaPipe.

    Drops alpha. Returns None for empty/invalid frames (never treat as current).
    """
    try:
        rgba = frame.convert(rtc.VideoBufferType.RGBA)
    except Exception as exc:  # noqa: BLE001
        logger.warning("video convert failed: %s", exc)
        return None
    w, h = rgba.width, rgba.height
    if w <= 0 or h <= 0:
        return None
    arr = np.frombuffer(rgba.data, dtype=np.uint8)
    expected = w * h * 4
    if arr.size < expected:
        return None
    rgba_img = arr[:expected].reshape((h, w, 4))
    rgb = rgba_img[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _pub_name(publication: rtc.RemoteTrackPublication) -> str:
    return (publication.name or publication.sid or "").strip()


class HermesCallWorker:
    """Owns recognition + captions for one LiveKit room / FaceTime call."""

    def __init__(self, ctx: JobContext, session: AgentSession) -> None:
        self.ctx = ctx
        self.session = session
        self.recognizer = SignRecognizer()
        self.captions: List[str] = []
        self.conversation: List[tuple[str, str]] = []
        self._latest_bgr: Optional[np.ndarray] = None
        self._busy = False
        self._video_stream: Optional[rtc.VideoStream] = None
        self._video_task: Optional[asyncio.Task] = None
        self._greeted = False
        self._tasks: List[asyncio.Task] = []
        self._speak_q: asyncio.Queue[str] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()

    def close(self) -> None:
        if self._video_stream is not None:
            self._video_stream.close()
            self._video_stream = None
        for task in list(self._tasks):
            task.cancel()
        self.recognizer.close()

    async def speak_loop(self) -> None:
        while True:
            text = await self._speak_q.get()
            await self._say(text)

    def attach_room_handlers(self) -> None:
        room = self.ctx.room

        @room.on("track_subscribed")
        def _on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            self._maybe_attach_video(track, publication)

        @room.on("track_unsubscribed")
        def _on_track_unsubscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if _pub_name(publication) == FACETIME_VIDEO:
                logger.info("facetime-remote-video unsubscribed — clearing stream")
                if self._video_stream is not None:
                    self._video_stream.close()
                    self._video_stream = None
                self._latest_bgr = None

        @room.on("room_metadata_changed")
        def _on_metadata(_old: str, new: str) -> None:
            if _bridge_ready(new):
                asyncio.create_task(self._greet_once())

        # Already-subscribed tracks at connect time.
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                track = publication.track
                if track is not None:
                    self._maybe_attach_video(track, publication)

        if _bridge_ready(room.metadata):
            asyncio.create_task(self._greet_once())

    def _maybe_attach_video(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        name = _pub_name(publication)
        if name != FACETIME_VIDEO:
            logger.debug("ignoring video track name=%r", name)
            return
        logger.info(
            "subscribing to %s (muted=%s)",
            FACETIME_VIDEO,
            getattr(publication, "muted", None),
        )
        if self._video_stream is not None:
            self._video_stream.close()
        if self._video_task is not None and not self._video_task.done():
            self._video_task.cancel()
        self._video_stream = rtc.VideoStream(track)
        self._video_task = asyncio.create_task(self._read_video(self._video_stream))
        self._tasks.append(self._video_task)

        def _cleanup(t: asyncio.Task) -> None:
            if t in self._tasks:
                self._tasks.remove(t)

        self._video_task.add_done_callback(_cleanup)

    async def _read_video(self, stream: rtc.VideoStream) -> None:
        try:
            async for event in stream:
                bgr = _video_frame_to_bgr(event.frame)
                if bgr is None:
                    continue
                self._latest_bgr = bgr
                if not self._busy:
                    asyncio.create_task(self._drain_frames())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("video stream ended: %s", exc)

    async def _drain_frames(self) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            while self._latest_bgr is not None:
                frame = self._latest_bgr
                self._latest_bgr = None
                await asyncio.to_thread(self._process_frame_sync, frame)
        finally:
            self._busy = False

    def _process_frame_sync(self, frame_bgr: np.ndarray) -> None:
        try:
            emitted, finished, _ = self.recognizer.process_frame(frame_bgr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("recognizer error: %s", exc)
            return
        if emitted:
            logger.info("sign token: %s (tokens=%s)", emitted, self.recognizer.tokens)
        if finished is None:
            return
        text = translate(finished.tokens, conversation=self.conversation)
        if not text:
            return
        logger.info("signer sentence: %s", text)
        self.conversation.append(("signer", text))
        self._loop.call_soon_threadsafe(self._speak_q.put_nowait, text)

    async def _say(self, text: str) -> None:
        try:
            await self.session.say(text, allow_interruptions=False)
        except Exception as exc:  # noqa: BLE001
            logger.error("session.say failed: %s", exc)

    async def _greet_once(self) -> None:
        if self._greeted:
            return
        self._greeted = True
        logger.info("chert_bridge_ready — greeting caller")
        await self._speak_q.put(GREETING)

    def on_transcript(self, ev: UserInputTranscribedEvent) -> None:
        if not ev.is_final:
            return
        text = (ev.transcript or "").strip()
        if not text:
            return
        self.captions.append(text)
        self.conversation.append(("reply", text))
        logger.info("caption [%d]: %s", len(self.captions), text)


async def entrypoint(ctx: JobContext) -> None:
    _require_runtime()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=elevenlabs.STT(),
        tts=elevenlabs.TTS(
            model=config.ELEVENLABS_TTS_MODEL,
            voice_id=config.ELEVENLABS_VOICE_ID,
        ),
        # No LLM — Hermes never answers on its own; only session.say() speaks.
    )

    worker = HermesCallWorker(ctx, session)
    speak_task = asyncio.create_task(worker.speak_loop())

    @session.on("user_input_transcribed")
    def _on_user_input(ev: UserInputTranscribedEvent) -> None:
        worker.on_transcript(ev)

    await ctx.connect()
    worker.attach_room_handlers()

    await session.start(
        agent=Agent(
            instructions=(
                "You are Hermes, a silent relay. Do not generate replies. "
                "Speech is only produced via explicit say() calls."
            )
        ),
        room=ctx.room,
    )

    async def _shutdown() -> None:
        speak_task.cancel()
        worker.close()

    ctx.add_shutdown_callback(_shutdown)


if __name__ == "__main__":
    # download-files / --help must work without LiveKit keys; start/dev need them.
    _cmd = next((a for a in sys.argv[1:] if not a.startswith("-")), "")
    if _cmd in ("", "start", "dev", "console"):
        _require_runtime()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=AGENT_NAME,
        )
    )
