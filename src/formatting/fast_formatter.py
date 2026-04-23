"""Fast rule-based formatter — runs on every transcript, <5ms target."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

_ENDING_PUNCTUATION = frozenset(".!?")


@dataclass
class FastFormatterResult:
    """Output from the fast formatter."""

    text: str
    latency_ms: float


def format_text(raw: str) -> FastFormatterResult:
    """Apply lightweight cleanup rules to raw STT output.

    Rules applied:
    - Strip leading/trailing whitespace
    - Capitalize the first character
    - Collapse runs of whitespace to a single space
    - Append a period if no sentence-ending punctuation is present

    Args:
        raw: Raw transcript string from Whisper.
    """
    t0 = time.perf_counter()
    text = raw.strip()
    text = re.sub(r"\s+", " ", text)
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in _ENDING_PUNCTUATION:
        text += "."
    elapsed = (time.perf_counter() - t0) * 1000
    return FastFormatterResult(text=text, latency_ms=elapsed)


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in text."""
    return len(text.split()) if text.strip() else 0
