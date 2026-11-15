"""ElevenLabs TTS (speak) and STT (transcribe)."""

from __future__ import annotations

import io
import re
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import requests

import config

TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"

# sounddevice's play()/rec() share one global stream: starting one while the
# other runs aborts it (PortAudio err -50). Serialize all audio I/O.
AUDIO_LOCK = threading.Lock()


def _require_key() -> str:
    if not config.ELEVENLABS_API_KEY:
        raise RuntimeError(
            "ELEVENLABS_API_KEY is missing. Add it to your .env file "
            "(see .env.example)."
        )
    return config.ELEVENLABS_API_KEY


def speak(text: str, voice_id: Optional[str] = None) -> None:
    """Synthesize speech with ElevenLabs and play it locally."""
    if not text.strip():
        return
    api_key = _require_key()
    vid = voice_id or config.ELEVENLABS_VOICE_ID
    url = TTS_URL.format(voice_id=vid)
    headers = {
        "xi-api-key": api_key,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": config.ELEVENLABS_TTS_MODEL,
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.7},
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs TTS failed ({resp.status_code}): {resp.text[:300]}"
        )

    with AUDIO_LOCK:
        _play(resp.content)


def _play(audio: bytes) -> None:
    # Prefer sounddevice + soundfile; fall back to writing a temp mp3 + afplay/ffplay.
    try:
        import soundfile as sf
        import sounddevice as sd

        data, sr = sf.read(io.BytesIO(audio))
        sd.play(data, sr)
        sd.wait()
        return
    except Exception:
        pass

    tmp = Path(tempfile.gettempdir()) / "hermes_tts.mp3"
    tmp.write_bytes(audio)
    # Best-effort local playback
    import shutil
    import subprocess

    for player in ("ffplay", "mpg123", "afplay"):
        if shutil.which(player):
            cmd = [player, "-nodisp", "-autoexit", str(tmp)] if player == "ffplay" else [player, str(tmp)]
            if player == "mpg123":
                cmd = [player, "-q", str(tmp)]
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    print(f"[voice] Saved TTS audio to {tmp} (no audio player found).")


def record_microphone(seconds: float = config.RECORD_SECONDS) -> np.ndarray:
    """Record mono audio from the default microphone."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is required to record the hearing person's reply. "
            "Install requirements.txt."
        ) from exc
    with AUDIO_LOCK:  # waits for any speech still playing
        print(f"[voice] Recording {seconds:.0f}s ...")
        audio = sd.rec(
            int(seconds * config.SAMPLE_RATE),
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    return audio.reshape(-1)


def _wav_bytes(audio: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


def transcribe(audio: Optional[np.ndarray] = None, seconds: float = config.RECORD_SECONDS) -> str:
    """
    Transcribe speech with ElevenLabs scribe_v1.
    If audio is None, records from the mic for `seconds`.
    """
    api_key = _require_key()
    if audio is None:
        audio = record_microphone(seconds=seconds)
    wav = _wav_bytes(audio)
    headers = {"xi-api-key": api_key}
    files = {"file": ("reply.wav", wav, "audio/wav")}
    data = {"model_id": config.ELEVENLABS_STT_MODEL}
    if config.ELEVENLABS_STT_LANGUAGE:
        data["language_code"] = config.ELEVENLABS_STT_LANGUAGE
    resp = requests.post(STT_URL, headers=headers, files=files, data=data, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs STT failed ({resp.status_code}): {resp.text[:300]}"
        )
    body = resp.json()
    text = str(body.get("text") or body.get("transcript") or "")
    # Drop sound-event tags such as "[pause]" or "(background noise)" so
    # silence does not show up as a reply caption.
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)
    return " ".join(text.split())
