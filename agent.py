"""Hermes call mode: LiveKit agent for Chert FaceTime.

Subscribes to Chert bridge tracks (`facetime-remote-video` /
`facetime-remote-audio`), runs the Phase 1 sign recognizer on caller video,
speaks translated English into the call via ElevenLabs TTS, and shows hearing
person captions on a published agent video board (FaceTime cannot draw its own
caption UI — text must ride on the agent's camera track).
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

# Spoken how-to (no separate mode UI on FaceTime — this is the menu).
GREETING = (
    "Hi, I'm Hermes, your sign language interpreter on this call. "
    "There is one mode: you sign, I speak your words out loud. "
    "When the other person talks, their words appear as large Reply text "
    "on my video on FaceTime. Watch that caption board, not Apple's captions."
)

# Portrait 9:16 — FaceTime on iPhone cover-crops landscape 16:9 and clips
# left-aligned labels. Keep all text inside SAFE_* insets for UI chrome.
BOARD_W = 720
BOARD_H = 1280
BOARD_FPS = 8
SAFE_X = 100      # ~14% side margin — survives iPhone cover + edge crop
SAFE_TOP = 110    # status / Dynamic Island
SAFE_BOTTOM = 160 # FaceTime call controls
CONTENT_W = BOARD_W - 2 * SAFE_X

_FONT_DIR = _ROOT / "assets" / "fonts"


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


def _load_font(name: str, size: int):
    """Load a vendored TTF; fall back to default bitmap font."""
    from PIL import ImageFont

    path = _FONT_DIR / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _fill_atmosphere(rgb) -> None:
    """Deep ink → teal wash with a warm amber glow (not purple/cream AI defaults)."""
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(rgb, "RGBA")
    for y in range(BOARD_H):
        t = y / max(BOARD_H - 1, 1)
        # top ink (12, 18, 28) → mid teal (18, 48, 58) → bottom charcoal (10, 14, 20)
        if t < 0.45:
            u = t / 0.45
            r = int(12 + (18 - 12) * u)
            g = int(18 + (48 - 18) * u)
            b = int(28 + (58 - 28) * u)
        else:
            u = (t - 0.45) / 0.55
            r = int(18 + (10 - 18) * u)
            g = int(48 + (14 - 48) * u)
            b = int(58 + (20 - 58) * u)
        draw.line([(0, y), (BOARD_W, y)], fill=(r, g, b, 255))
    # Soft amber glow behind brand (atmospheric, not a card)
    glow = Image.new("RGBA", (BOARD_W, BOARD_H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    cx, cy, rad = BOARD_W // 2, SAFE_TOP + 40, 220
    for i in range(6, 0, -1):
        a = 18 if i == 6 else max(6, 28 - i * 4)
        gdraw.ellipse(
            [cx - rad * i // 6, cy - rad * i // 8, cx + rad * i // 6, cy + rad * i // 8],
            fill=(212, 160, 72, a),
        )
    rgb.alpha_composite(glow)


def _rounded_rect(
    draw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int, int],
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def _text_width(font, text: str) -> int:
    try:
        return int(font.getlength(text))
    except Exception:  # noqa: BLE001
        bbox = font.getbbox(text)
        return int(bbox[2] - bbox[0])


def _wrap_px(text: str, font, max_width: int, max_lines: int) -> List[str]:
    text = (text or "").strip() or "—"
    words = text.split()
    lines: List[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        if _text_width(font, trial) <= max_width:
            cur = trial
            continue
        if cur:
            lines.append(cur)
        cur = word
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if not lines:
        lines = ["—"]
    # Ellipsize last line if we still overflow
    if len(lines) == max_lines:
        last = lines[-1]
        while last and _text_width(font, last) > max_width:
            last = last[:-1]
        if last != lines[-1]:
            lines[-1] = (last[:-1] + "…") if len(last) > 1 else "…"
    return lines


def render_caption_board(
    signer_text: str,
    reply_text: str,
    tokens: str = "",
) -> np.ndarray:
    """BGR portrait board for FaceTime — safe margins, readable YOU/REPLY."""
    from PIL import Image, ImageDraw

    rgb = Image.new("RGBA", (BOARD_W, BOARD_H), (12, 18, 28, 255))
    _fill_atmosphere(rgb)
    draw = ImageDraw.Draw(rgb)

    brand = _load_font("NotoSerifDisplay-Bold.ttf", 72)
    tagline = _load_font("NotoSansDisplay-Regular.ttf", 22)
    label = _load_font("NotoSansDisplay-Bold.ttf", 26)
    body = _load_font("NotoSansDisplay-Regular.ttf", 36)
    reply_body = _load_font("NotoSansDisplay-Bold.ttf", 44)
    meta = _load_font("NotoSansDisplay-Regular.ttf", 20)

    x0 = SAFE_X
    y = SAFE_TOP

    # Brand — hero signal (must survive without nav chrome)
    brand_text = "HERMES"
    bw = _text_width(brand, brand_text)
    draw.text(((BOARD_W - bw) // 2, y), brand_text, font=brand, fill=(245, 232, 210, 255))
    y += 82
    tip = "You sign · I speak · replies below"
    tw = _text_width(tagline, tip)
    draw.text(((BOARD_W - tw) // 2, y), tip, font=tagline, fill=(168, 196, 188, 255))
    y += 48

    # Accent rule
    draw.rectangle(
        [BOARD_W // 2 - 36, y, BOARD_W // 2 + 36, y + 3],
        fill=(212, 160, 72, 220),
    )
    y += 28

    pad = 36
    panel_w = CONTENT_W

    # YOU panel (signer) — secondary
    you_h = 260
    _rounded_rect(
        draw,
        (x0, y, x0 + panel_w, y + you_h),
        22,
        (16, 36, 40, 210),
    )
    draw.text((x0 + pad, y + 26), "YOU", font=label, fill=(120, 210, 170, 255))
    draw.text(
        (x0 + pad + 78, y + 30),
        "signer",
        font=meta,
        fill=(140, 170, 160, 255),
    )
    you_lines = _wrap_px(signer_text, body, panel_w - 2 * pad, max_lines=3)
    ty = y + 78
    for line in you_lines:
        draw.text((x0 + pad, ty), line, font=body, fill=(245, 248, 246, 255))
        ty += 46
    if tokens:
        tok = f"Signs: {tokens}"
        while _text_width(meta, tok) > panel_w - 2 * pad and len(tok) > 8:
            tok = tok[:-2]
        draw.text((x0 + pad, y + you_h - 42), tok, font=meta, fill=(130, 150, 145, 255))
    y += you_h + 32

    # REPLY panel — primary for Deaf viewer (largest type)
    reply_bottom = BOARD_H - SAFE_BOTTOM
    reply_h = max(360, reply_bottom - y)
    _rounded_rect(
        draw,
        (x0, y, x0 + panel_w, y + reply_h),
        22,
        (22, 42, 62, 230),
    )
    # Warm left rail for hierarchy (well inside safe margin)
    draw.rectangle(
        [x0 + 18, y + 28, x0 + 24, y + reply_h - 28],
        fill=(212, 160, 72, 230),
    )
    draw.text((x0 + pad + 12, y + 28), "REPLY", font=label, fill=(180, 220, 255, 255))
    draw.text(
        (x0 + pad + 118, y + 32),
        "hearing person",
        font=meta,
        fill=(140, 175, 200, 255),
    )
    reply_lines = _wrap_px(reply_text, reply_body, panel_w - 2 * pad - 16, max_lines=5)
    ry = y + 84
    for line in reply_lines:
        draw.text((x0 + pad + 12, ry), line, font=reply_body, fill=(255, 255, 255, 255))
        ry += 56

    bgr = cv2.cvtColor(np.array(rgb.convert("RGB")), cv2.COLOR_RGB2BGR)
    return bgr


def _bgr_to_rgba_bytes(bgr: np.ndarray) -> bytes:
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    return rgba.tobytes()


class CaptionBoard:
    """Publishes a camera track so FaceTime can show reply captions."""

    def __init__(self) -> None:
        self.signer_text = "Waiting for your signs..."
        self.reply_text = "Waiting for the other person to speak..."
        self.tokens = ""
        self._source = rtc.VideoSource(BOARD_W, BOARD_H)
        self._track = rtc.LocalVideoTrack.create_video_track("hermes-captions", self._source)
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self, room: rtc.Room) -> None:
        opts = rtc.TrackPublishOptions()
        opts.source = rtc.TrackSource.SOURCE_CAMERA
        await room.local_participant.publish_track(self._track, opts)
        self._running = True
        self._push_frame()
        self._task = asyncio.create_task(self._loop())
        logger.info("published hermes-captions video board (%dx%d @ %dfps)", BOARD_W, BOARD_H, BOARD_FPS)

    def update(
        self,
        *,
        signer: Optional[str] = None,
        reply: Optional[str] = None,
        tokens: Optional[str] = None,
    ) -> None:
        if signer is not None:
            self.signer_text = signer
        if reply is not None:
            self.reply_text = reply
        if tokens is not None:
            self.tokens = tokens

    def _push_frame(self) -> None:
        bgr = render_caption_board(self.signer_text, self.reply_text, self.tokens)
        frame = rtc.VideoFrame(
            BOARD_W,
            BOARD_H,
            rtc.VideoBufferType.RGBA,
            _bgr_to_rgba_bytes(bgr),
        )
        self._source.capture_frame(frame)

    async def _loop(self) -> None:
        try:
            while self._running:
                self._push_frame()
                await asyncio.sleep(1.0 / BOARD_FPS)
        except asyncio.CancelledError:
            raise

    async def close(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        try:
            await self._source.aclose()
        except Exception:  # noqa: BLE001
            pass


class HermesCallWorker:
    """Owns recognition + captions for one LiveKit room / FaceTime call."""

    def __init__(self, ctx: JobContext, session: AgentSession, board: CaptionBoard) -> None:
        self.ctx = ctx
        self.session = session
        self.board = board
        # Prefer the shipped classifier; if load fails (format mismatch), keep
        # the call alive for greeting + reply captions without sign video.
        try:
            self.recognizer: Optional[SignRecognizer] = SignRecognizer()
        except Exception as exc:  # noqa: BLE001
            logger.error("classifier load failed — signs disabled: %s", exc)
            self.recognizer = None
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
        if self.recognizer is not None:
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
        if self.recognizer is None:
            return
        try:
            emitted, finished, _ = self.recognizer.process_frame(frame_bgr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("recognizer error: %s", exc)
            return
        if emitted:
            logger.info("sign token: %s (tokens=%s)", emitted, self.recognizer.tokens)
            token_line = " ".join(self.recognizer.tokens)
            self._loop.call_soon_threadsafe(
                lambda: self.board.update(tokens=token_line)
            )
        if finished is None:
            return
        text = translate(finished.tokens, conversation=self.conversation)
        if not text:
            return
        logger.info("signer sentence: %s", text)
        self.conversation.append(("signer", text))
        self._loop.call_soon_threadsafe(
            lambda t=text: self.board.update(signer=t, tokens="")
        )
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
        self.board.update(
            signer="Listening for your signs…",
            reply="Waiting for the other person to speak...",
        )
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
        self.board.update(reply=text)


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

    board = CaptionBoard()
    worker = HermesCallWorker(ctx, session, board)
    speak_task = asyncio.create_task(worker.speak_loop())

    @session.on("user_input_transcribed")
    def _on_user_input(ev: UserInputTranscribedEvent) -> None:
        worker.on_transcript(ev)

    await ctx.connect()
    # Publish caption video from the same participant that will speak (Chert rule).
    await board.start(ctx.room)
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
        await board.close()
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
