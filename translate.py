"""Gemini translation: ASL gloss tokens → one natural English sentence."""

from __future__ import annotations

from typing import Sequence

from google import genai
from google.genai import types

import config

SYSTEM_PROMPT = (
    "You convert ASL gloss into English. The input is a sequence of recognized "
    "sign language tokens (ASL gloss), possibly with recognition errors. "
    "Output ONE short, natural, first-person English sentence. "
    "Do not invent facts that are not supported by the signs or conversation context. "
    "If the signs are unclear, produce the most conservative short sentence you can."
)


def _api_key() -> str:
    key = config.GEMINI_API_KEY
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is missing. "
            "Add it to your .env file (see .env.example)."
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
        print(f"[translate] {exc} — using token fallback.")
        return fallback

    parts: list[str] = []
    if conversation:
        ctx_lines = [f"{role}: {text}" for role, text in conversation[-6:]]
        parts.append("Recent conversation:\n" + "\n".join(ctx_lines))
    parts.append("ASL gloss tokens: " + " ".join(clean))
    user_prompt = "\n\n".join(parts)

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=80,
            ),
        )
        text = (resp.text or "").strip()
        if "\n" in text:
            text = text.splitlines()[0].strip()
        return text or fallback
    except Exception as exc:  # noqa: BLE001
        print(f"[translate] Gemini API failed ({exc}) — using token fallback.")
        return fallback
