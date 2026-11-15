"""Grok (xAI) translation: ASL gloss tokens → one natural English sentence."""

from __future__ import annotations

from typing import List, Sequence

from openai import OpenAI

import config

SYSTEM_PROMPT = (
    "You convert ASL gloss into English. The input is a sequence of recognized "
    "sign language tokens (ASL gloss), possibly with recognition errors. "
    "Output ONE short, natural, first-person English sentence. "
    "Do not invent facts that are not supported by the signs or conversation context. "
    "If the signs are unclear, produce the most conservative short sentence you can."
)


def _client() -> OpenAI:
    if not config.XAI_API_KEY:
        raise RuntimeError(
            "XAI_API_KEY is missing. Add it to your .env file "
            "(see .env.example)."
        )
    return OpenAI(api_key=config.XAI_API_KEY, base_url=config.XAI_BASE_URL)


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
        client = _client()
    except RuntimeError as exc:
        print(f"[translate] {exc} — using token fallback.")
        return fallback

    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation:
        ctx_lines = []
        for role, text in conversation[-6:]:
            ctx_lines.append(f"{role}: {text}")
        messages.append(
            {
                "role": "user",
                "content": "Recent conversation:\n" + "\n".join(ctx_lines),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": "ASL gloss tokens: " + " ".join(clean),
        }
    )

    try:
        resp = client.chat.completions.create(
            model=config.GROK_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=80,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Keep a single sentence if the model rambles.
        if "\n" in text:
            text = text.splitlines()[0].strip()
        return text or fallback
    except Exception as exc:  # noqa: BLE001
        print(f"[translate] Grok API failed ({exc}) — using token fallback.")
        return fallback
