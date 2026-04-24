"""Fast rule-based formatter — runs on every transcript, <5ms target."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

_TERMINAL_PUNCTUATION = frozenset(".!?,;:")

# Multi-word phrases must come before their component single words so the
# alternation matches the longest token first.
_FILLERS = (
    "you know", "i mean", "kind of", "sort of",
    "umm", "uhh", "um", "uh",
    "basically", "literally", "actually",
    "like", "so", "right",
)

_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(f) for f in _FILLERS) + r")\b",
    re.IGNORECASE,
)


class FastFormatter:
    """Lightweight, zero-dependency transcript formatter with filler-word stripping."""

    def format(self, text: str) -> str:
        """Strip fillers, normalise whitespace, capitalise, append period if needed."""
        text = _FILLER_RE.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        text = text[0].upper() + text[1:]
        if text[-1] not in _TERMINAL_PUNCTUATION:
            text += "."
        return text


@dataclass
class FastFormatterResult:
    """Output from the fast formatter module-level helper."""

    text: str
    latency_ms: float


_formatter = FastFormatter()


def format_text(raw: str) -> FastFormatterResult:
    """Apply FastFormatter rules and return a timed result dataclass."""
    t0 = time.perf_counter()
    text = _formatter.format(raw)
    return FastFormatterResult(text=text, latency_ms=(time.perf_counter() - t0) * 1000)


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in text."""
    return len(text.split()) if text.strip() else 0
