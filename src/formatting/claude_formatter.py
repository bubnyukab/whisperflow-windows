"""Claude API formatter — optional paid formatter using the Anthropic SDK."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.config.settings import Settings

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a transcription editor. Fix grammar, punctuation, and formatting of the "
    "spoken text below. Preserve the speaker's meaning exactly. "
    "Output only the corrected text, nothing else."
)


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
    api_key = settings.get_anthropic_api_key()
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
