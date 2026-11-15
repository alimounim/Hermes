"""Gemini translation: ASL gloss tokens → one natural English sentence."""

from __future__ import annotations

from typing import Sequence

from google import genai
from google.genai import types

import config

SYSTEM_PROMPT = """You are Hermes, a sign-language interpreter for a Deaf/hard-of-hearing person in a hospital, pharmacy, or airport setting.

The input is ASL gloss: a short sequence of recognized sign tokens (possibly noisy or out of order). Your job is to weave those tokens into ONE short, natural English sentence — a fluent question OR statement — that a hearing person would understand out loud.

Known gloss vocabulary (ignore unknowns conservatively):
_none (ignore), HELLO, THANK-YOU, PLEASE, YES, NO, WHERE, HEAD, PAIN, SICK,
BATHROOM, WATER, HUNGRY, THIRSTY, NEED, WAIT, FINISH, BYE, FINE, YOU, MY,
HOME, HELP, TIME, NOW, I, MEDICINE, PHARMACY,
DRINK, FOOD, SLEEP, SLEEPY, HOT, BATH, BED, BAD, BETTER, HAPPY, SAD, CRY,
WHO, WHY, GO, STAY, GIVE, HAVE, LOOK, LISTEN, FIND, LATER, TOMORROW,
YESTERDAY, MORNING, NIGHT, NOT, ANOTHER, MOM, DAD, PERSON.
Note: NOT negates the sign next to it (NOT HUNGRY → "I'm not hungry").
ANOTHER means "another / one more". BAD + a body word usually means it feels bad.
Note: PAIN means "it hurts / owie", BATHROOM covers toilet/restroom, NEED covers
"have to / must". Prefer hospital/pharmacy phrasing when MEDICINE, PHARMACY,
PAIN, SICK, HEAD, HELP, WATER, THIRSTY, or HUNGRY appear.

Rules:
- Output exactly ONE sentence. No quotes, no labels, no explanation, no bullet points.
- Use fluent grammar and function words (is/are, the, a, please, where, I, need, etc.) so the result sounds spoken, not like raw gloss.
- Prefer first person when the signer is talking about themselves (I / MY).
- Questions are fine when the gloss implies one (WHERE, HELP, NEED often → question).
- Hospital / pharmacy / travel context is OK for small connecting words only.
- Do NOT invent people, places, times, symptoms, or objects that are not suggested by the tokens or recent conversation.
- If tokens are unclear, stay conservative: still make a grammatical sentence, but do not add new facts.
- Translate ONLY the current gloss tokens. Recent conversation is context, not content: never repeat or merge in things the signer said earlier, and do not add words like "still" or "also" that refer back to earlier turns.
- Use the conversation only to resolve a short reply to the hearing person's last question (e.g. reply "Do you need water?" + gloss YES → "Yes, I do.").
- Repeated tokens (PLEASE PLEASE) mean the sign was emphasized or detected twice: say it once.
- Every token must be reflected in the sentence. Never drop a token because it is awkward: TIME HELLO FINISH YES → "Hello, yes, is it time to finish?", not "Hello, is it time now?".
- The signer may put words in ASL order (topic first, question word last: BATHROOM WHERE → "Where is the bathroom?"). Reorder freely into natural English.

Examples (gloss → English):
- WHERE MEDICINE PAIN → Where is the pain medicine?
- I NEED HELP → I need help.
- WHERE BATHROOM → Where is the bathroom?
- THANK-YOU → Thank you.
- I NEED MEDICINE → I need medicine.
- HELLO I NEED MEDICINE HEAD PAIN → Hello, I need medicine for a headache.
- HELLO → Hello.
- PLEASE HELP → Please help me.
- HEAD PAIN → My head hurts.
- WHERE PHARMACY → Where is the pharmacy?
- I THIRSTY NEED WATER → I am thirsty and need water.
- I HUNGRY → I am hungry.
- I SICK NEED HELP → I feel sick and need help.
- WAIT PLEASE → Please wait.
- TIME NOW → What time is it now?
- MY HOME → My home.
- BYE → Goodbye.
- FINE → I am fine.
- YES → Yes.
- NO → No.
- I PAIN HEAD NEED MEDICINE → I have a headache and need medicine.
- HELP WHERE BATHROOM → Can you help me find the bathroom?
"""


def _api_key() -> str:
    key = config.GEMINI_API_KEY
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is missing. "
            "Add it to your .env file (see .env.example). "
            "Without it, Hermes only joins raw sign tokens."
        )
    return key


def tokens_to_fallback(tokens: Sequence[str]) -> str:
    return " ".join(t.replace("-", " ").title() for t in tokens if t and t != "_none")


def translate(
    tokens: Sequence[str],
    conversation: Sequence[tuple[str, str]] | None = None,
) -> str:
    """
    Translate sign tokens into one English sentence.

    conversation: optional list of (role, text) turns, e.g.
      [("signer", "I need help."), ("reply", "How long has it hurt?")]
    so answers fit the dialogue ("THREE DAY" → "For three days.").
    """
    clean = [t for t in tokens if t and t != "_none"]
    if not clean:
        return ""

    fallback = tokens_to_fallback(clean)
    try:
        api_key = _api_key()
    except RuntimeError as exc:
        print(f"[translate] {exc} — using token fallback: {fallback!r}")
        return fallback

    parts: list[str] = []
    if conversation:
        ctx_lines = [f"{role}: {text}" for role, text in conversation[-6:]]
        parts.append(
            "Recent conversation (context only — do not repeat it):\n"
            + "\n".join(ctx_lines)
        )
    parts.append("Current ASL gloss tokens to translate: " + " ".join(clean))
    parts.append("English sentence:")
    user_prompt = "\n\n".join(parts)

    client = _client(api_key)
    # Try the configured model, then faster fallbacks. Busy models (503/504)
    # fail fast instead of stalling the conversation.
    models = list(dict.fromkeys([config.GEMINI_MODEL, *config.GEMINI_FALLBACK_MODELS]))
    for model in models:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    # Headroom for models that spend output tokens on thinking.
                    max_output_tokens=512,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[translate] {model} failed: {str(exc)[:120]}")
            continue
        text = _clean(resp.text or "")
        if text:
            print(f"[translate] Gemini ({model}): {text!r}")
            return text
        print(f"[translate] {model} returned empty text")
    print(f"[translate] all Gemini models failed — using token fallback: {fallback!r}")
    return fallback


_CLIENT: genai.Client | None = None


def _client(api_key: str) -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=config.GEMINI_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    return _CLIENT


def _clean(text: str) -> str:
    """Strip accidental quotes / labels / extra lines from a model reply."""
    text = text.strip()
    if "\n" in text:
        text = text.splitlines()[0].strip()
    if text.lower().startswith("english:"):
        text = text.split(":", 1)[1].strip()
    if len(text) > 1 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1].strip()
    return text
