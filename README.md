# Hermes

Sign-language translator for HackUTA 2026. Hermes watches a signer through a camera, recognizes signs, turns them into natural English with Google Gemini, and speaks them with ElevenLabs. When a hearing person replies, Hermes transcribes their speech and shows captions.

**Phase 1 (this PR):** core engine + in-person mode. Call mode (`agent.py`) comes in Phase 2.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your keys
```

### Environment variables

| Variable | Required for | Notes |
|---|---|---|
| `GEMINI_API_KEY` | Translation | Google AI Studio / Gemini key (`GOOGLE_API_KEY` also accepted) |
| `GEMINI_MODEL` | Translation | Default `gemini-3.5-flash-lite` (fast); falls back to `GEMINI_FALLBACK_MODELS` (default `gemini-3.8-flash`) |
| `ELEVENLABS_API_KEY` | Speak + reply captions | ElevenLabs key |
| `ELEVENLABS_VOICE_ID` | Speak | Default voice id in `.env.example` |

The MediaPipe hand and body-pose `.task` models download automatically into `models/` on first run.

> Phase 1 no longer uses xAI / Grok. You can remove any old `XAI_API_KEY` from `.env`.

## How signs are recognized

One sign at a time: **raise your hands, make the sign, lower your hands.** Hermes classifies the whole movement once your hands drop below the dotted "rest line" (about elbow height) or leave the picture. It looks at hand shape, where the hands are relative to your face and shoulders, and how they move.

- Confident (default ≥ 65%) → the sign is added to the sentence.
- Not confident, or not a sign it knows → it shows "Not sure" with its two best guesses and adds nothing. Sign it again.
- 3 seconds with no new sign → the sentence is translated (Gemini) and spoken (ElevenLabs).

## Build the model

1. Download the Kaggle competition data ([asl-signs](https://www.kaggle.com/competitions/asl-signs)) and unpack it to `data/asl-signs/`.
2. Extract the landmarks for the Hermes signs (seconds):

```bash
python build_dataset.py
```

3. Train (a few minutes). This first prints an honest accuracy check on signers the model never saw, then trains on everything and writes `models/sign_clf.joblib`:

```bash
python train.py
```

### Add your own signs

The dataset has no **I**, **YOU**, **HELP**, **MEDICINE** or **PHARMACY**. Record them (about 15 each), then retrain:

```bash
python record_signs.py I HELP MEDICINE PHARMACY YOU
python train.py --no-eval
```

## Run in-person mode

```bash
python app.py
```

The top line always says what Hermes is doing: *Ready*, *Signing…*, *Added: WATER (94%)* or *Not sure (…) - try again*. The picture is mirrored so it is easier to sign at.

- **U** — undo the last sign
- **S** — send the sentence now instead of waiting
- **R** — record the hearing person’s reply for 5 seconds and caption it
- **C** — clear captions
- **Q** — quit

Options:

```bash
python app.py --no-voice          # skip TTS
python app.py --video demo.mp4    # backup demo from a recorded clip
```

Tuning in `.env`: `HERMES_CONFIDENCE` (default 0.65; higher = fewer wrong words, more retries), `HERMES_SENTENCE_GAP` (default 3.0 s), `HERMES_REST_LINE` (default 1.6 shoulder-widths below the nose), `HERMES_SIGN_END` (default 0.35 s with hands down).

## Vocabulary

49 signs from the dataset:

HELLO, THANK-YOU, PLEASE, YES, NO, NOT, ANOTHER, WHERE, WHO, WHY, HEAD, PAIN, SICK, BAD, BETTER, HOT, BATHROOM, WATER, DRINK, FOOD, HUNGRY, THIRSTY, NEED, SLEEP, SLEEPY, BED, BATH, HAVE, WAIT, FINISH, STAY, LISTEN, FIND, HAPPY, SAD, CRY, FINE, BYE, TIME, NOW, LATER, TOMORROW, YESTERDAY, MORNING, NIGHT, MY, MOM, DAD, HOME.

The mapping from dataset sign names is `SIGN_MAP` in `import_gislr.py`. GO, LOOK, GIVE and PERSON are mapped but left out (`EXCLUDED_SIGNS` in `config.py`) because unseen signers got them right under half the time.

## Project layout

| File | Role |
|---|---|
| `config.py` | Settings, paths, env keys |
| `build_dataset.py` | Kaggle parquet files → `data/gislr_clips.npz` |
| `signfeat.py` | Whole-sign features (shape, place, movement) shared by training and live |
| `train.py` | Accuracy check on unseen signers + train the classifier |
| `record_signs.py` | Record your own signs for words the dataset lacks |
| `landmarks.py` | MediaPipe hand + body tracking |
| `recognizer.py` | Cuts the camera stream into signs → sentences |
| `sources.py` | Webcam / video file frames |
| `translate.py` | Gemini gloss → English |
| `voice.py` | ElevenLabs TTS / STT |
| `app.py` | In-person mode UI |
| `collect.py`, `import_gislr.py` | Older per-frame pipeline (not used by the current model) |

## Troubleshooting

- **Camera missing:** `Could not open camera index 0` — close other apps using the webcam, or change `CAMERA_INDEX` in `config.py`.
- **No classifier / "old-style model":** run `build_dataset.py` then `train.py` before `app.py`.
- **Signs never start or never end:** make sure your face and shoulders are in the picture, and lower your hands below the dotted line between signs.
- **Missing API keys:** Hermes prints a clear error and falls back (token join for translation; skips speak/transcribe if ElevenLabs is unset).
