#!/usr/bin/env python3
"""Render caption-board previews + FaceTime cover-crop checks (no LiveKit)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for mod in (
    "livekit",
    "livekit.rtc",
    "livekit.agents",
    "livekit.plugins",
    "livekit.plugins.elevenlabs",
    "livekit.plugins.silero",
    "recognizer",
    "translate",
    "config",
):
    sys.modules[mod] = MagicMock()
sys.modules["config"].ELEVENLABS_API_KEY = "x"
sys.modules["config"].CLASSIFIER_PATH = ROOT / "models" / "sign_clf.joblib"

import cv2  # noqa: E402
import agent  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/cursor/artifacts")
    out.mkdir(parents=True, exist_ok=True)

    bgr = agent.render_caption_board(
        "Waiting for your signs...",
        "Waiting for the other person to speak...",
        tokens="HELLO WORLD",
    )
    assert bgr.shape == (agent.BOARD_H, agent.BOARD_W, 3), bgr.shape
    cv2.imwrite(str(out / "caption-board-hd.png"), bgr)

    phone, sx, _sy = agent._cover_crop_view(bgr, 390, 844)
    cv2.imwrite(str(out / "caption-board-iphone.png"), phone)

    mac, _mx, my = agent._cover_crop_view(bgr, 1280, 720)
    cv2.imwrite(str(out / "caption-board-mac.png"), mac)

    tile, _, _ = agent._cover_crop_view(bgr, 200, 356)
    cv2.imwrite(str(out / "caption-board-tile.png"), tile)

    safe_x = int(agent.BOARD_W * agent.SAFE_X_FRAC)
    stack_top = int(agent.BOARD_H * agent.STACK_TOP_FRAC)
    stack_bot = int(agent.BOARD_H * agent.STACK_BOTTOM_FRAC)
    text_left = safe_x + 36

    # iPhone cover: side crop must leave text inset.
    assert sx + 24 < text_left, f"iPhone side crop {sx} eats text at {text_left}"

    # Mac landscape cover: visible source band must contain the stack.
    scale = max(1280 / agent.BOARD_W, 720 / agent.BOARD_H)
    vis_h = int(round(720 / scale))
    vis_y0, vis_y1 = my, my + vis_h
    assert vis_y0 <= stack_top + 24, f"stack top {stack_top} above Mac visible {vis_y0}"
    assert vis_y1 >= stack_bot - 24, f"stack bot {stack_bot} below Mac visible {vis_y1}"

    print(
        f"ok board={agent.BOARD_W}x{agent.BOARD_H}@{agent.RENDER_SCALE}x "
        f"iphone_crop_x={sx} mac_vis_y={vis_y0}-{vis_y1} "
        f"stack={stack_top}-{stack_bot} -> {out}"
    )


if __name__ == "__main__":
    main()
