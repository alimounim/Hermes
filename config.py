"""Hermes configuration: vocabulary, thresholds, paths, and env keys."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SAMPLES_DIR = DATA_DIR / "samples"
CLASSIFIER_PATH = MODELS_DIR / "sign_clf.joblib"
HAND_LANDMARKER_MODEL = MODELS_DIR / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Vocabulary — edit freely. `_none` is rest / no sign.
# GISLR imports: PAIN←owie, BATHROOM←potty, NEED←haveto. I/MEDICINE/PHARMACY
# are kept for Ali's own recordings (not in the Kaggle set).
VOCABULARY = [
    "_none",
    "HELLO",
    "THANK-YOU",
    "PLEASE",
    "YES",
    "NO",
    "WHERE",
    "HEAD",
    "PAIN",
    "SICK",
    "BATHROOM",
    "WATER",
    "HUNGRY",
    "THIRSTY",
    "NEED",
    "WAIT",
    "FINISH",
    "BYE",
    "FINE",
    "YOU",
    "MY",
    "HOME",
    "HELP",
    "TIME",
    "NOW",
    "I",
    "MEDICINE",
    "PHARMACY",
]

# Landmark / feature settings
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
FEATURE_DIM = NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 126
WINDOW_SIZE = 15

# Recognition smoothing
CONFIDENCE_THRESHOLD = 0.75
CONSECUTIVE_FRAMES = 6
SENTENCE_GAP_SEC = 1.5

# Training
SAMPLES_PER_SIGN = 30
MLP_HIDDEN = (128, 64)
MLP_MAX_ITER = 500

# Camera / UI
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Env keys — translation uses Google Gemini (XAI/Grok no longer required for Phase 1)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Tried in order when GEMINI_MODEL is busy or fails (comma-separated in .env).
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.8-flash").split(",")
    if m.strip()
]
# Gemini rejects deadlines under 10s.
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "10000"))

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")
ELEVENLABS_STT_MODEL = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")

RECORD_SECONDS = 5
SAMPLE_RATE = 16000
