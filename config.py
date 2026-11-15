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
    "DRINK",
    "FOOD",
    "SLEEP",
    "SLEEPY",
    "HOT",
    "BATH",
    "BED",
    "BAD",
    "BETTER",
    "HAPPY",
    "SAD",
    "CRY",
    "WHO",
    "WHY",
    "GO",
    "STAY",
    "GIVE",
    "HAVE",
    "LOOK",
    "LISTEN",
    "FIND",
    "LATER",
    "TOMORROW",
    "YESTERDAY",
    "MORNING",
    "NIGHT",
    "NOT",
    "ANOTHER",
    "MOM",
    "DAD",
    "PERSON",
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

# Sentences
# Seconds with no new sign before the sentence is translated and spoken.
SENTENCE_GAP_SEC = float(os.getenv("HERMES_SENTENCE_GAP", "3.0"))

# Training (whole-sign model; see train.py). Chosen by testing on signers the
# model never saw: 4 x (512, 256) voting, strong weight decay, PCA to 400.
MLP_HIDDEN = (512, 256)
MLP_ALPHA = 0.1
MLP_MAX_ITER = 80
PCA_COMPONENTS = 400
ENSEMBLE_SIZE = 4
AUG_PER_CLIP = 4

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
# Language of the hearing person's reply. Auto-detect misfires on short/quiet
# clips (e.g. returns Portuguese), so pin it. Empty string = auto-detect.
ELEVENLABS_STT_LANGUAGE = os.getenv("ELEVENLABS_STT_LANGUAGE", "en")

RECORD_SECONDS = 5
SAMPLE_RATE = 16000

# Whole-sign recognition (segment model)
# Pose landmarks kept alongside the hands: nose, shoulders, elbows, wrists.
POSE_KEEP = [0, 11, 12, 13, 14, 15, 16]
POSE_LANDMARKER_MODEL = MODELS_DIR / "pose_landmarker_lite.task"
POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
GISLR_CLIPS = DATA_DIR / "gislr_clips.npz"
CUSTOM_SIGNS_DIR = DATA_DIR / "custom_signs"  # your own recordings (record_signs.py)

# A sign = hands come up, sign, hands go down. Hands lower than this many
# shoulder-widths below the nose (about elbow height) count as "down", so
# resting them on the desk or your lap ends the sign.
REST_BELOW_NOSE = float(os.getenv("HERMES_REST_LINE", "1.6"))
# Seconds with hands down before the sign is judged.
SIGN_END_SEC = float(os.getenv("HERMES_SIGN_END", "0.35"))
# Seconds with no hands detected at all before the sign is judged. Longer,
# because the tracker can lose a fast-moving hand for a moment mid-sign.
SIGN_LOST_SEC = float(os.getenv("HERMES_SIGN_LOST", "0.7"))
# Shortest / longest movement treated as one sign.
SIGN_MIN_FRAMES = 6
SIGN_MAX_SEC = 6.0
# Below this confidence the sign is shown as "not sure" and NOT added.
# At 0.65 the model is right ~91% of the time on unseen signers and accepts
# ~5 signs in 6; raise it for fewer wrong words, lower it for fewer retries.
SIGN_CONFIDENCE = float(os.getenv("HERMES_CONFIDENCE", "0.65"))

# Legacy (old per-frame pipeline: collect.py / import_gislr.py window samples).
SAMPLES_PER_SIGN = 30

# Signs left out of the model because unseen signers got them right under half
# the time (they mostly turned into other signs). Remove a name to bring it back.
EXCLUDED_SIGNS = {"GO", "LOOK", "GIVE", "PERSON"}
