"""Claude API formatter — optional paid formatter using the Anthropic SDK."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.config.settings import Settings

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a voice transcription formatter. The user dictated text using "
    "speech-to-text. Your job:\n"
    "1. Remove filler words (um, uh, like, you know)\n"
    "2. Fix grammar, punctuation, capitalisation\n"
    "3. Infer the user's actual intent — e.g. 'send email to sarah about meeting' "
    "becomes 'Send an email to Sarah about the meeting.'\n"
    "4. Preserve tone (casual stays casual, formal stays formal)\n"
    "5. Format as a list if it sounds like one\n"
    "Return ONLY the formatted text. No explanation, no preamble, no quotes."
)


class ClaudeFormatter:
    """Formats voice transcripts via the Claude API."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", timeout: float = 10.0) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def format(self, raw_transcript: str, context_hint: str = "") -> str:
        """Send transcript to Claude API and return cleaned text.

        Falls back to raw_transcript on any error — never raises.
        """
        import anthropic

        system = _SYSTEM_PROMPT
        if context_hint:
            system = f"{system}\n\nContext: {context_hint}"
        try:
            client = anthropic.Anthropic(api_key=self._api_key, timeout=self._timeout)
            message = client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": raw_transcript}],
            )
            text = message.content[0].text.strip()
            return text or raw_transcript
        except Exception:
            log.exception("ClaudeFormatter.format failed")
            return raw_transcript

    def is_available(self) -> bool:
        """Return True if an API key is configured (no network call)."""
        return bool(self._api_key)


@dataclass
class ClaudeFormatterResult:
    """Output from the Claude API formatter."""

    text: str
    latency_ms: float
    model: str
    success: bool


def format_text(raw: str, settings: Settings) -> ClaudeFormatterResult:
    """Send raw transcript to the Claude API and return the formatted result.

    Requires ANTHROPIC_API_KEY in environment. Falls back gracefully on any error.

    Args:
        raw: Raw or fast-formatted transcript.
        settings: App settings (provides get_anthropic_api_key()).
    """
    t0 = time.perf_counter()
    api_key = settings.anthropic_api_key
    if not api_key:
        elapsed = (time.perf_counter() - t0) * 1000
        log.warning("Claude formatter called but ANTHROPIC_API_KEY is not set")
        return ClaudeFormatterResult(text=raw, latency_ms=elapsed, model="none", success=False)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"{_SYSTEM_PROMPT}\n\n{raw}"}],
        )
        text = message.content[0].text.strip()
        elapsed = (time.perf_counter() - t0) * 1000
        return ClaudeFormatterResult(
            text=text or raw,
            latency_ms=elapsed,
            model=message.model,
            success=bool(text),
        )
    except Exception:
        elapsed = (time.perf_counter() - t0) * 1000
        log.exception("Claude formatter failed")
        return ClaudeFormatterResult(text=raw, latency_ms=elapsed, model="claude", success=False)
