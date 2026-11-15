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
import time
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

# FaceTime fit/contain: a locked 9:16 board pillarboxes into a thin Mac strip
# (facetime-blurry-retest.png) and looks soft when upscaled. Default publish is
# Full-HD landscape; we flip to portrait when the remote FaceTime camera is
# portrait so iPhone fills the screen too.
BOARD_LANDSCAPE = (1920, 1080)
BOARD_PORTRAIT = (1080, 1920)
BOARD_W, BOARD_H = BOARD_LANDSCAPE  # default; CaptionBoard may swap at runtime
# Low fps → more bits per frame for static captions (screen-share style).
BOARD_FPS = 5
# Encode budget for sharp glyphs; FaceTime re-encodes again so leave headroom.
BOARD_MAX_BITRATE = 8_000_000
# Re-push identical frame occasionally so bridge/FaceTime keep the picture;
# do NOT stream every tick — that defeats screencast zero-hertz and adds mush.
BOARD_KEEPALIVE_SEC = 2.0
# Render at 2× then downsample for glyph shapes; sharpen after.
RENDER_SCALE = 2

# Title-safe fractions (of published frame). Tuned for FaceTime fit + mild
# cover crop in scripts/preview_caption_board.py.
SAFE_X_FRAC_LANDSCAPE = 0.05
SAFE_X_FRAC_PORTRAIT = 0.10
SAFE_TOP_FRAC = 0.06
SAFE_BOTTOM_FRAC = 0.07
# Landscape: brand + side-by-side panels. Portrait: stacked YOU then REPLY.
STACK_TOP_FRAC_LANDSCAPE = 0.08
STACK_BOTTOM_FRAC_LANDSCAPE = 0.93
STACK_TOP_FRAC_PORTRAIT = 0.12
STACK_BOTTOM_FRAC_PORTRAIT = 0.90

# Back-compat aliases for preview script / fractional helpers (landscape default).
SAFE_X_FRAC = SAFE_X_FRAC_LANDSCAPE
STACK_TOP_FRAC = STACK_TOP_FRAC_LANDSCAPE
STACK_BOTTOM_FRAC = STACK_BOTTOM_FRAC_LANDSCAPE

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
        return ImageFont.truetype(str(path), size=max(1, int(size)))
    except OSError:
        return ImageFont.load_default()


def _px(frac: float, axis: str, scale: int = 1, *, w: int = BOARD_W, h: int = BOARD_H) -> int:
    base = w if axis == "x" else h
    return int(round(frac * base * scale))


def _sharpen_bgr(bgr: np.ndarray) -> np.ndarray:
    """Unsharp mask so H.264 / FaceTime re-encode keep glyph edges."""
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=0.9)
    sharp = cv2.addWeighted(bgr, 1.45, blur, -0.45, 0)
    # Mild contrast boost on luma — helps thin strokes survive compression.
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.convertScaleAbs(l, alpha=1.08, beta=4)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _fill_atmosphere(rgb, w: int, h: int) -> None:
    """Flat ink background.

    Soft gradients/glows steal H.264 bits from text edges and look like mosquito
    noise after FaceTime re-encodes the bridge feed — keep the plate flat.
    """
    from PIL import Image

    rgb.paste(Image.new("RGBA", (w, h), (14, 24, 30, 255)))


def _draw_text(
    draw,
    xy: tuple[int, int],
    text: str,
    font,
    fill: tuple[int, int, int, int],
    *,
    stroke: int = 0,
    stroke_fill: tuple[int, int, int, int] = (0, 0, 0, 255),
) -> None:
    """Draw text with an optional dark outline (survives lossy re-encode)."""
    if stroke > 0:
        draw.text(
            xy,
            text,
            font=font,
            fill=fill,
            stroke_width=stroke,
            stroke_fill=stroke_fill,
        )
    else:
        draw.text(xy, text, font=font, fill=fill)


def _fit_letterbox_view(
    bgr: np.ndarray, view_w: int, view_h: int
) -> tuple[np.ndarray, int, int, int, int]:
    """Simulate FaceTime object-fit:contain (black bars). Returns view + pad."""
    h, w = bgr.shape[:2]
    scale = min(view_w / w, view_h / h)
    sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((view_h, view_w, 3), dtype=np.uint8)
    x0 = max(0, (view_w - sw) // 2)
    y0 = max(0, (view_h - sh) // 2)
    canvas[y0 : y0 + sh, x0 : x0 + sw] = resized
    return canvas, x0, y0, sw, sh


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
    if len(lines) == max_lines:
        last = lines[-1]
        while last and _text_width(font, last) > max_width:
            last = last[:-1]
        if last != lines[-1]:
            lines[-1] = (last[:-1] + "…") if len(last) > 1 else "…"
    return lines


def _cover_crop_view(
    bgr: np.ndarray, view_w: int, view_h: int
) -> tuple[np.ndarray, int, int]:
    """Simulate FaceTime object-fit:cover. Returns (crop, src_x0, src_y0)."""
    h, w = bgr.shape[:2]
    scale = max(view_w / w, view_h / h)
    sw, sh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    x0 = max(0, (sw - view_w) // 2)
    y0 = max(0, (sh - view_h) // 2)
    crop = resized[y0 : y0 + view_h, x0 : x0 + view_w]
    # Map crop origin back to source pixels.
    src_x0 = int(round(x0 / scale))
    src_y0 = int(round(y0 / scale))
    return crop, src_x0, src_y0


def render_caption_board(
    signer_text: str,
    reply_text: str,
    tokens: str = "",
    *,
    width: int = BOARD_W,
    height: int = BOARD_H,
) -> np.ndarray:
    """BGR HD board — landscape YOU|REPLY or portrait stack, retina-sharp."""
    from PIL import Image, ImageDraw

    portrait = height >= width
    s = RENDER_SCALE
    rw, rh = width * s, height * s
    rgb = Image.new("RGBA", (rw, rh), (14, 24, 30, 255))
    _fill_atmosphere(rgb, rw, rh)
    draw = ImageDraw.Draw(rgb)

    safe_x_frac = SAFE_X_FRAC_PORTRAIT if portrait else SAFE_X_FRAC_LANDSCAPE
    stack_top_frac = STACK_TOP_FRAC_PORTRAIT if portrait else STACK_TOP_FRAC_LANDSCAPE
    stack_bot_frac = STACK_BOTTOM_FRAC_PORTRAIT if portrait else STACK_BOTTOM_FRAC_LANDSCAPE

    safe_x = _px(safe_x_frac, "x", s, w=width, h=height)
    safe_top = _px(SAFE_TOP_FRAC, "y", s, w=width, h=height)
    stack_top = _px(stack_top_frac, "y", s, w=width, h=height)
    stack_bot = _px(stack_bot_frac, "y", s, w=width, h=height)
    content_w = rw - 2 * safe_x
    gap = max(18 * s, content_w // 48)
    pad = max(28 * s, content_w // 36)
    radius = 18 * s
    short = min(rw, rh)
    # Outline width scales with render resolution — keeps edges after downsample.
    stroke_lg = max(3 * s, int(short * 0.006))
    stroke_sm = max(2 * s, int(short * 0.004))

    # Large bold type — FaceTime re-encode destroys thin glyphs.
    brand = _load_font("NotoSerifDisplay-Bold.ttf", short * (0.095 if portrait else 0.100))
    tagline = _load_font("NotoSansDisplay-Bold.ttf", short * 0.036)
    label = _load_font("NotoSansDisplay-Bold.ttf", short * 0.044)
    body = _load_font("NotoSansDisplay-Bold.ttf", short * (0.058 if portrait else 0.068))
    reply_body = _load_font("NotoSansDisplay-Bold.ttf", short * (0.062 if portrait else 0.072))
    meta = _load_font("NotoSansDisplay-Bold.ttf", short * 0.034)

    y = max(safe_top, stack_top)
    brand_text = "HERMES"
    bw = _text_width(brand, brand_text)
    _draw_text(
        draw,
        ((rw - bw) // 2, y),
        brand_text,
        brand,
        (255, 248, 230, 255),
        stroke=stroke_lg,
        stroke_fill=(0, 0, 0, 255),
    )
    y += int(short * 0.105)
    tip = "You sign · I speak · REPLY listens"
    tw = _text_width(tagline, tip)
    _draw_text(
        draw,
        ((rw - tw) // 2, y),
        tip,
        tagline,
        (200, 230, 220, 255),
        stroke=stroke_sm,
        stroke_fill=(0, 0, 0, 255),
    )
    y += int(short * 0.052)
    rule_w = int(rw * 0.045)
    draw.rectangle(
        [rw // 2 - rule_w, y, rw // 2 + rule_w, y + max(4, 4 * s)],
        fill=(212, 160, 72, 255),
    )
    y += int(short * 0.028)

    panel_top = y
    panel_bot = stack_bot
    panel_h = max(1, panel_bot - panel_top)

    def _draw_you(x0: int, top: int, pw: int, ph: int, max_lines: int) -> None:
        # Opaque flat panels — no translucent mush for the encoder.
        _rounded_rect(draw, (x0, top, x0 + pw, top + ph), radius, (18, 42, 46, 255))
        _draw_text(
            draw,
            (x0 + pad, top + pad // 2),
            "YOU",
            label,
            (120, 230, 180, 255),
            stroke=stroke_sm,
        )
        you_label_w = _text_width(label, "YOU")
        _draw_text(
            draw,
            (x0 + pad + you_label_w + 14 * s, top + pad // 2 + 4 * s),
            "signer",
            meta,
            (180, 210, 200, 255),
            stroke=max(1, stroke_sm // 2),
        )
        you_lines = _wrap_px(signer_text, body, pw - 2 * pad, max_lines=max_lines)
        line_gap = int(short * 0.078)
        ty = top + pad // 2 + int(short * 0.078)
        for line in you_lines:
            _draw_text(
                draw,
                (x0 + pad, ty),
                line,
                body,
                (255, 255, 255, 255),
                stroke=stroke_lg,
            )
            ty += line_gap
        if tokens:
            tok = f"Signs: {tokens}"
            while _text_width(meta, tok) > pw - 2 * pad and len(tok) > 8:
                tok = tok[:-2]
            _draw_text(
                draw,
                (x0 + pad, top + ph - pad // 2 - int(short * 0.040)),
                tok,
                meta,
                (170, 200, 190, 255),
                stroke=max(1, stroke_sm // 2),
            )

    def _draw_reply(x0: int, top: int, pw: int, ph: int, max_lines: int) -> None:
        _rounded_rect(draw, (x0, top, x0 + pw, top + ph), radius, (24, 48, 72, 255))
        rail = max(6 * s, pw // 60)
        draw.rectangle(
            [x0 + pad // 2, top + pad // 2, x0 + pad // 2 + rail, top + ph - pad // 2],
            fill=(212, 160, 72, 255),
        )
        tx = x0 + pad // 2 + rail + 12 * s
        _draw_text(
            draw,
            (tx, top + pad // 2),
            "REPLY",
            label,
            (190, 230, 255, 255),
            stroke=stroke_sm,
        )
        reply_label_w = _text_width(label, "REPLY")
        _draw_text(
            draw,
            (tx + reply_label_w + 14 * s, top + pad // 2 + 4 * s),
            "hearing person",
            meta,
            (170, 205, 230, 255),
            stroke=max(1, stroke_sm // 2),
        )
        reply_lines = _wrap_px(
            reply_text, reply_body, pw - pad - rail - 20 * s, max_lines=max_lines
        )
        reply_gap = int(short * 0.082)
        ry = top + pad // 2 + int(short * 0.082)
        for line in reply_lines:
            _draw_text(
                draw,
                (tx, ry),
                line,
                reply_body,
                (255, 255, 255, 255),
                stroke=stroke_lg,
            )
            ry += reply_gap

    if portrait:
        # Stacked panels — fills iPhone FaceTime without side pillarbox mush.
        you_h = int(panel_h * 0.42)
        reply_h = panel_h - you_h - gap
        _draw_you(safe_x, panel_top, content_w, you_h, max_lines=3)
        _draw_reply(safe_x, panel_top + you_h + gap, content_w, reply_h, max_lines=4)
    else:
        # Side-by-side — fills Mac FaceTime landscape window.
        panel_w = (content_w - gap) // 2
        _draw_you(safe_x, panel_top, panel_w, panel_h, max_lines=4)
        _draw_reply(safe_x + panel_w + gap, panel_top, panel_w, panel_h, max_lines=4)

    # Downsample 2× → publish size, then sharpen for the lossy FaceTime path.
    rgb_full = rgb.convert("RGB")
    if s != 1:
        # AREA downsample keeps edges harder than LANCZOS for video text.
        arr = np.array(rgb_full)
        bgr_hi = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr_hi, (width, height), interpolation=cv2.INTER_AREA)
    else:
        bgr = cv2.cvtColor(np.array(rgb_full), cv2.COLOR_RGB2BGR)
    return _sharpen_bgr(bgr)


def _bgr_to_rgba_bytes(bgr: np.ndarray) -> bytes:
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    return rgba.tobytes()


class CaptionBoard:
    """Publishes a camera track so FaceTime can show reply captions."""

    def __init__(self) -> None:
        self.signer_text = "Waiting for your signs..."
        self.reply_text = "Waiting for the other person to speak..."
        self.tokens = ""
        self.width, self.height = BOARD_LANDSCAPE
        self._room: Optional[rtc.Room] = None
        self._source: Optional[rtc.VideoSource] = None
        self._track: Optional[rtc.LocalVideoTrack] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._adapt_lock = asyncio.Lock()
        self._dirty = True
        self._cached_rgba: Optional[bytes] = None
        self._cache_key: Optional[tuple] = None
        self._last_push = 0.0
        self._build_source()

    def _build_source(self) -> None:
        # Screencast hint: keep resolution under congestion, no denoise mush,
        # zero-hertz when the board is static (see dirty/keepalive loop).
        self._source = rtc.VideoSource(self.width, self.height, is_screencast=True)
        self._track = rtc.LocalVideoTrack.create_video_track(
            "hermes-captions", self._source
        )
        self._cached_rgba = None
        self._cache_key = None
        self._dirty = True

    def _publish_opts(self) -> rtc.TrackPublishOptions:
        return rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,  # Chert requires camera, not screenshare
            # Single high-quality layer — avoid a low simulcast tier on FaceTime.
            simulcast=False,
            video_codec=rtc.VideoCodec.H264,
            # Software encode often keeps text edges better than HW defaults.
            video_encoder=rtc.VideoEncoderBackend.ENCODER_BACKEND_SOFTWARE,
            video_encoding=rtc.VideoEncoding(
                max_bitrate=BOARD_MAX_BITRATE,
                max_framerate=float(BOARD_FPS),
            ),
            degradation_preference=rtc.DegradationPreference.MAINTAIN_RESOLUTION,
        )

    async def _publish(self, room: rtc.Room) -> None:
        assert self._track is not None
        pub = await room.local_participant.publish_track(
            self._track, self._publish_opts()
        )
        # Confirm LiveKit accepted the source dimensions (not a tiny default).
        pub_w = getattr(pub, "width", None) or 0
        pub_h = getattr(pub, "height", None) or 0
        logger.info(
            "published hermes-captions board source=%dx%d pub=%dx%d "
            "fps=%d bitrate=%dkbit screencast=1 simulcast=%s mime=%s",
            self.width,
            self.height,
            pub_w,
            pub_h,
            BOARD_FPS,
            BOARD_MAX_BITRATE // 1000,
            getattr(pub, "simulcasted", None),
            getattr(pub, "mime_type", None),
        )
        if pub_w and pub_h and (pub_w < self.width * 0.9 or pub_h < self.height * 0.9):
            logger.warning(
                "LiveKit publication smaller than source (%dx%d < %dx%d) — "
                "FaceTime will look soft if the bridge keeps this layer",
                pub_w,
                pub_h,
                self.width,
                self.height,
            )

    async def start(self, room: rtc.Room) -> None:
        self._room = room
        await self._publish(room)
        self._running = True
        self._push_frame(force=True)
        self._task = asyncio.create_task(self._loop())

    async def adapt_to_remote_frame(self, frame_w: int, frame_h: int) -> None:
        """Match board aspect to the caller's FaceTime camera (Mac vs iPhone)."""
        if frame_w <= 0 or frame_h <= 0 or self._room is None:
            return
        want_portrait = frame_h >= int(frame_w * 1.15)
        target = BOARD_PORTRAIT if want_portrait else BOARD_LANDSCAPE
        if target == (self.width, self.height):
            return
        async with self._adapt_lock:
            if target == (self.width, self.height) or self._room is None:
                return
            room = self._room
            old_track = self._track
            old_source = self._source
            old_sid = getattr(old_track, "sid", None) if old_track is not None else None
            self.width, self.height = target
            self._build_source()
            assert self._track is not None
            try:
                if old_sid:
                    await room.local_participant.unpublish_track(old_sid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("unpublish old caption board failed: %s", exc)
            try:
                if old_source is not None:
                    await old_source.aclose()
            except Exception:  # noqa: BLE001
                pass
            await self._publish(room)
            self._push_frame(force=True)
            logger.info(
                "adapted hermes-captions board to %dx%d (remote %dx%d)",
                self.width,
                self.height,
                frame_w,
                frame_h,
            )

    def update(
        self,
        *,
        signer: Optional[str] = None,
        reply: Optional[str] = None,
        tokens: Optional[str] = None,
    ) -> None:
        changed = False
        if signer is not None and signer != self.signer_text:
            self.signer_text = signer
            changed = True
        if reply is not None and reply != self.reply_text:
            self.reply_text = reply
            changed = True
        if tokens is not None and tokens != self.tokens:
            self.tokens = tokens
            changed = True
        if changed:
            self._dirty = True

    def _push_frame(self, *, force: bool = False) -> None:
        if self._source is None:
            return
        key = (self.signer_text, self.reply_text, self.tokens, self.width, self.height)
        if (
            not force
            and self._cached_rgba is not None
            and self._cache_key == key
        ):
            rgba = self._cached_rgba
        else:
            bgr = render_caption_board(
                self.signer_text,
                self.reply_text,
                self.tokens,
                width=self.width,
                height=self.height,
            )
            # Sanity: published pixels must match the LiveKit source size.
            assert bgr.shape[1] == self.width and bgr.shape[0] == self.height, bgr.shape
            rgba = _bgr_to_rgba_bytes(bgr)
            self._cached_rgba = rgba
            self._cache_key = key
        frame = rtc.VideoFrame(
            self.width,
            self.height,
            rtc.VideoBufferType.RGBA,
            rgba,
        )
        self._source.capture_frame(frame)
        self._last_push = time.monotonic()
        self._dirty = False

    async def _loop(self) -> None:
        """Push on content change; keepalive rarely so screencast zero-hertz works."""
        try:
            while self._running:
                now = time.monotonic()
                due_keepalive = (now - self._last_push) >= BOARD_KEEPALIVE_SEC
                if self._dirty or due_keepalive:
                    self._push_frame(force=due_keepalive and not self._dirty)
                await asyncio.sleep(1.0 / max(BOARD_FPS, 1))
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
        if self._room is not None and self._track is not None:
            sid = getattr(self._track, "sid", None)
            if sid:
                try:
                    await self._room.local_participant.unpublish_track(sid)
                except Exception:  # noqa: BLE001
                    pass
        if self._source is not None:
            try:
                await self._source.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._source = None


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
        self._last_remote_wh: Optional[tuple[int, int]] = None

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
                # Match caption board aspect to the caller's camera (Mac vs phone).
                h, w = bgr.shape[:2]
                wh = (w, h)
                if wh != self._last_remote_wh:
                    self._last_remote_wh = wh
                    asyncio.create_task(self.board.adapt_to_remote_frame(w, h))
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
