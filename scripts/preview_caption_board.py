#!/usr/bin/env python3
"""Render caption-board previews + FaceTime fit/cover checks (no LiveKit)."""

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

    lw, lh = agent.BOARD_LANDSCAPE
    pw, ph = agent.BOARD_PORTRAIT

    land = agent.render_caption_board(
        "Listening for your signs…",
        "Waiting for the other person to speak...",
        tokens="HELLO WORLD",
        width=lw,
        height=lh,
    )
    port = agent.render_caption_board(
        "Listening for your signs…",
        "Waiting for the other person to speak...",
        tokens="HELLO WORLD",
        width=pw,
        height=ph,
    )
    assert land.shape == (lh, lw, 3), land.shape
    assert port.shape == (ph, pw, 3), port.shape
    assert lw > lh, "landscape board must be wider than tall"
    assert ph > pw, "portrait board must be taller than wide"

    cv2.imwrite(str(out / "caption-board-sharp.png"), land)
    cv2.imwrite(str(out / "caption-board-hd.png"), land)
    cv2.imwrite(str(out / "caption-board-portrait.png"), port)

    # FaceTime Mac: fit/contain into landscape window (matches blurry-retest.png mode).
    mac_fit, mx, my, mw, mh = agent._fit_letterbox_view(land, 1280, 720)
    cv2.imwrite(str(out / "caption-board-mac.png"), mac_fit)
    fill_ratio = (mw * mh) / float(1280 * 720)
    assert fill_ratio > 0.95, f"Mac fit fill too low: {fill_ratio:.3f}"
    assert mx < 40 and my < 40, f"Mac fit unexpected bars mx={mx} my={my}"

    # Old portrait board on Mac — document how bad the pillarbox was.
    mac_old, omx, omy, omw, omh = agent._fit_letterbox_view(port, 1280, 720)
    cv2.imwrite(str(out / "caption-board-mac-portrait-fit.png"), mac_old)
    old_fill = (omw * omh) / float(1280 * 720)
    assert old_fill < 0.45, f"portrait-on-Mac should be thin, fill={old_fill:.3f}"

    # FaceTime iPhone: portrait board fills; landscape letterboxes.
    phone_port, ppx, ppy, ppw, pph = agent._fit_letterbox_view(port, 390, 844)
    cv2.imwrite(str(out / "caption-board-iphone.png"), phone_port)
    phone_fill = (ppw * pph) / float(390 * 844)
    assert phone_fill > 0.55, f"iPhone portrait fill too low: {phone_fill:.3f}"

    phone_land, _, _, plw, plh = agent._fit_letterbox_view(land, 390, 844)
    cv2.imwrite(str(out / "caption-board-iphone-landscape-fit.png"), phone_land)

    phone_cover, sx, _sy = agent._cover_crop_view(port, 390, 844)
    cv2.imwrite(str(out / "caption-board-iphone-cover.png"), phone_cover)
    mac_cover, _mx2, my2 = agent._cover_crop_view(land, 1280, 720)
    cv2.imwrite(str(out / "caption-board-mac-cover.png"), mac_cover)

    tile, _, _ = agent._cover_crop_view(land, 320, 180)
    cv2.imwrite(str(out / "caption-board-tile.png"), tile)

    print(
        f"ok land={lw}x{lh} port={pw}x{ph} @{agent.RENDER_SCALE}x "
        f"bitrate={agent.BOARD_MAX_BITRATE} fps={agent.BOARD_FPS} "
        f"mac_land_fill={fill_ratio:.3f} mac_port_fill={old_fill:.3f} "
        f"iphone_port_fill={phone_fill:.3f} "
        f"iphone_cover_x={sx} mac_cover_y={my2} -> {out}"
    )


if __name__ == "__main__":
    main()
