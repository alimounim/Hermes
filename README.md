# Hermes

Sign-language translator built at HackUTA 2026. Hermes watches a signer through a laptop camera, recognizes ASL signs, turns them into natural English with Google Gemini, and speaks them with ElevenLabs. When the hearing person replies, Hermes transcribes their speech and shows it as a caption, so the conversation goes both ways.

It is built for short, practical exchanges at a pharmacy, clinic or front desk: about 50 everyday signs such as HELLO, PAIN, HEAD, SICK, BATHROOM, WATER, HUNGRY, WAIT and THANK-YOU.

## Try it now: [signtovoice.us](https://signtovoice.us)

No install needed. From an iPhone, iPad or Mac:

1. Open **[signtovoice.us](https://signtovoice.us)** and tap **FaceTime Hermes**, or start a FaceTime **Video** call to **+1 (702) 510-9706**.
2. Sign one word at a time (raise your hands, sign, lower them), then pause for 3 seconds. Hermes says your sentence out loud.
3. When someone speaks back, their words appear under **REPLY** on Hermes' video.

Don't know ASL? The **[sign dictionary](https://signtovoice.us/dictionary.html)** has a video for each of the 49 signs and shows which ones Hermes recognizes most reliably. Good first tries: HELLO, BATHROOM → WHERE, THIRSTY → WATER → PLEASE.

The line takes one call at a time and is online during HackUTA 2026 demos. If it doesn't answer, try again in a few minutes.

## Quick start

Needs Python 3.11–3.13, a webcam, and API keys for Gemini and ElevenLabs. The trained sign model is included, so no dataset download is needed to run it.

```bash
git clone https://github.com/alimounim/Hermes.git
cd Hermes
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then add your keys
python app.py
```

On macOS, allow camera and microphone access for your terminal app when asked. The MediaPipe hand and body-pose models (about 13 MB) download into `models/` on first run.

### Environment variables

| Variable | Required for | Notes |
|---|---|---|
| `GEMINI_API_KEY` | Translation | Google AI Studio / Gemini key (`GOOGLE_API_KEY` also accepted) |
| `GEMINI_MODEL` | Translation | Default `gemini-3.5-flash-lite` (fast); falls back to `GEMINI_FALLBACK_MODELS` (default `gemini-3.8-flash`) |
| `ELEVENLABS_API_KEY` | Speak + reply captions | ElevenLabs key |
| `ELEVENLABS_VOICE_ID` | Speak | Default voice id in `.env.example` |

## How signs are recognized

One sign at a time: **raise your hands, make the sign, lower your hands.** Hermes classifies the whole movement once your hands drop below the dotted "rest line" (about elbow height) or leave the picture. It looks at hand shape, where the hands are relative to your face and shoulders, and how they move.

- Confident (default ≥ 65%) → the sign is added to the sentence.
- Not confident, or not a sign it knows → it shows "Not sure" with its two best guesses and adds nothing. Sign it again.
- 3 seconds with no new sign → the sentence is translated (Gemini) and spoken (ElevenLabs).

## Rebuild the model (optional)

Only needed to change the vocabulary or retrain. The included `models/sign_clf.joblib` was trained this way.

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
| `agent.py` | Call mode (Chert FaceTime via LiveKit) |

## Call mode (Chert FaceTime)

1. Put keys in `.env` / `.env.local` (never chat): `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` (same LiveKit project), `ELEVENLABS_API_KEY`, optional `GEMINI_API_KEY`, `AGENT_NAME=hermes-agent`.
2. Classifier ships at `models/sign_clf.joblib` (tracked in git). If it is missing locally, run `git checkout origin/main -- models/sign_clf.joblib`, or rebuild it with `python train.py`.
3. Terminal A — start the agent worker:

```bash
git pull
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

**What FaceTime call mode does today (one mode):** you sign → Hermes **speaks** English into the call. Hearing-person speech is transcribed and shown as **Reply** text on Hermes’ video board: 1920×1080 landscape by default, switching to 1080×1920 portrait when the caller's camera is portrait (iPhone). The board streams at a steady 15 fps; a slower stream shows up as a gray tile in FaceTime (Apple FaceTime has no agent caption API — watch Hermes’ video pane). Layout uses fractional safe margins + a centered YOU/REPLY stack that survives phone and Mac FaceTime cover-crop. There is no in-call mode picker UI; the spoken greeting is the how-to.

To keep the line answering without your laptop, run both on a server: see [deploy/README.md](deploy/README.md).

Trial note: Chert-managed LiveKit (`chert livekit`) is not provisioned on this trial — use your own LiveKit Cloud project.

## Troubleshooting

- **Camera missing:** `Could not open camera index 0` — close other apps using the webcam, or change `CAMERA_INDEX` in `config.py`.
- **No classifier / "old-style model":** run `build_dataset.py` then `train.py` before `app.py`.
- **Signs never start or never end:** make sure your face and shoulders are in the picture, and lower your hands below the dotted line between signs.
- **Missing API keys:** Hermes prints a clear error and falls back (token join for translation; skips speak/transcribe if ElevenLabs is unset).
- **LiveKit not set:** `chert doctor` fails on livekit until `.env.local` has all three `LIVEKIT_*` values from one project.
