"""Gemini translation: ASL gloss tokens → one natural English sentence."""

from __future__ import annotations

from typing import Sequence

from google import genai
from google.genai import types

import config

SYSTEM_PROMPT = """You are Hermes, a sign-language interpreter for a Deaf/hard-of-hearing person in a hospital, pharmacy, or airport setting.

The input is ASL gloss: a short sequence of recognized sign tokens (possibly noisy or out of order). Your job is to weave those tokens into ONE short, natural English sentence — a fluent question OR statement — that a hearing person would understand out loud.

Rules:
- Output exactly ONE sentence. No quotes, no labels, no explanation, no bullet points.
- Use fluent grammar and function words (is/are, the, a, please, where, I, need, etc.) so the result sounds spoken, not like raw gloss.
- Prefer first person when the signer is talking about themselves (I / my).
- Questions are fine when the gloss implies one (WHERE, HELP, NEED often → question).
- Hospital / pharmacy / travel context is OK for small connecting words only.
- Do NOT invent people, places, times, symptoms, or objects that are not suggested by the tokens or recent conversation.
- If tokens are unclear, stay conservative: still make a grammatical sentence, but do not add new facts.
- If recent conversation is provided, fit the answer to that dialogue when the gloss is a reply.

Examples (gloss → English):
- WHERE MEDICINE PAIN → Where is the pain medicine?
- I NEED HELP → I need help.
- WHERE BATHROOM → Where is the bathroom?
- THANK-YOU → Thank you.
- I NEED MEDICINE → I need medicine.
- HELLO → Hello.
- PLEASE HELP → Please help me.
- HEAD PAIN → My head hurts.
- WHERE PHARMACY → Where is the pharmacy?
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
        parts.append("Recent conversation:\n" + "\n".join(ctx_lines))
    parts.append("ASL gloss tokens: " + " ".join(clean))
    parts.append("English sentence:")
    user_prompt = "\n\n".join(parts)

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=80,
            ),
        )
        text = (resp.text or "").strip()
        # Strip accidental quotes / labels
        if text.startswith(("\"", "'")) and text.endswith(("\"", "'")) and len(text) > 1:
            text = text[1:-1].strip()
        if "\n" in text:
            text = text.splitlines()[0].strip()
        if text.lower().startswith("english:"):
            text = text.split(":", 1)[1].strip()
        if not text:
            print(f"[translate] empty Gemini response — fallback: {fallback!r}")
            return fallback
        print(f"[translate] Gemini ({config.GEMINI_MODEL}): {text!r}")
        return text
    except Exception as exc:  # noqa: BLE001
        print(f"[translate] Gemini API failed ({exc}) — using token fallback: {fallback!r}")
        return fallback
