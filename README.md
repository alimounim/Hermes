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
| `GEMINI_MODEL` | Translation | Default `gemini-3.8-flash` |
| `ELEVENLABS_API_KEY` | Speak + reply captions | ElevenLabs key |
| `ELEVENLABS_VOICE_ID` | Speak | Default voice id in `.env.example` |

The MediaPipe HandLandmarker `.task` model downloads automatically into `models/` on first run.

> Phase 1 no longer uses xAI / Grok. You can remove any old `XAI_API_KEY` from `.env`.

## Collect training data

Stand in front of the webcam. For each sign, hold the pose/motion and press **SPACE** to save a sample. Press **N** for the next sign, **Q** to quit. Aim for ~30 samples per sign (including `_none` = hands resting).

```bash
python collect.py
```

Samples are saved under `data/samples/<SIGN>/*.npy`.

## Train the classifier

```bash
python train.py
```

Prints a classification report and writes `models/sign_clf.joblib`.

## Run in-person mode

```bash
python app.py
```

- Live camera with landmarks, current sign + confidence, collected tokens
- Caption panel: `Signer: …` / `Reply: …`
- When a sentence finishes (1.5s idle after signs): translate with Gemini, speak with ElevenLabs
- **R** — record the hearing person’s reply for 5 seconds and caption it
- **C** — clear captions
- **Q** — quit

Options:

```bash
python app.py --no-voice          # skip TTS
python app.py --video demo.mp4    # backup demo from a recorded clip
```

## Vocabulary

Configured in `config.py` (must stay aligned with `import_gislr.SIGN_MAP` targets):

`_none`, HELLO, THANK-YOU, PLEASE, YES, NO, WHERE, HEAD, PAIN, SICK, BATHROOM, WATER, HUNGRY, THIRSTY, NEED, WAIT, FINISH, BYE, FINE, YOU, MY, HOME, HELP, TIME, NOW, I, MEDICINE, PHARMACY.

GISLR covers everything except **I**, **MEDICINE**, and **PHARMACY** — record those yourself with `collect.py` (or leave them out of training until you do).

## Project layout

| File | Role |
|---|---|
| `config.py` | Vocab, thresholds, paths, env keys |
| `landmarks.py` | MediaPipe Tasks HandLandmarker + features |
| `sources.py` | Webcam / video file frames |
| `collect.py` | Record training samples |
| `train.py` | Train scikit-learn MLP |
| `recognizer.py` | Smoothed signs → sentences |
| `translate.py` | Gemini gloss → English |
| `voice.py` | ElevenLabs TTS / STT |
| `app.py` | In-person mode UI |
| `agent.py` | Call mode (Chert FaceTime via LiveKit) |

## Call mode (Chert FaceTime)

1. Put keys in `.env` / `.env.local` (never chat): `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` (same LiveKit project), `ELEVENLABS_API_KEY`, optional `GEMINI_API_KEY`, `AGENT_NAME=hermes-agent`.
2. Classifier ships at `models/sign_clf.joblib` (tracked in git). If missing locally: `git checkout origin/main -- models/sign_clf.joblib`, or `python train.py` / `python scripts/make_smoke_classifier.py`.
3. Terminal A — start the agent worker:

```bash
git pull origin cursor/hermes-phase2-facetime-24b1
source .venv/bin/activate
python agent.py start
```

4. Terminal B — connect Chert to this project/line:

```bash
npx --yes --prefer-online @trychert/facetime@0.1.0-beta.21 setup --env-file .env.local --project c849395a-f541-42ad-81a3-a12d9fa43d94
npx --yes --prefer-online @trychert/facetime@0.1.0-beta.21 doctor
npx --yes --prefer-online @trychert/facetime@0.1.0-beta.21 dev --agent hermes-agent
```

5. From an iPhone: FaceTime **Video** to **+17025109706** (or text `join Q7UYAY` to that number once if joining from another phone). You should hear Hermes greet within ~10s, then spoken English when signs are recognized.

Trial note: Chert-managed LiveKit (`chert livekit`) is not provisioned on this trial — use your own LiveKit Cloud project.

## Troubleshooting

- **Camera missing:** `Could not open camera index 0` — close other apps using the webcam, or change `CAMERA_INDEX` in `config.py`.
- **No classifier:** run `collect.py` then `train.py` before `app.py`.
- **Missing API keys:** Hermes prints a clear error and falls back (token join for translation; skips speak/transcribe if ElevenLabs is unset).
- **LiveKit not set:** `chert doctor` fails on livekit until `.env.local` has all three `LIVEKIT_*` values from one project.
