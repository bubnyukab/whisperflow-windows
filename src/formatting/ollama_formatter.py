"""Ollama LLM formatter — calls local Ollama API to clean up transcripts."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from src.config.settings import Settings

log = logging.getLogger(__name__)

# Completion-style prompt — model fills in "Output:" lines, never enters chat/answer mode.
_COMPLETION_PROMPT_TEMPLATE = """\
### TRANSCRIPT CLEANING TASK
For each Input line, write ONLY the cleaned transcript on the Output line.
Rules: remove filler words (um, uh, like, you know), fix capitalisation, add punctuation.
Do NOT answer questions. Do NOT explain anything. Do NOT add any words. Just copy and clean.

Input: um so i was thinking uh maybe we could go to the store tomorrow you know
Output: So I was thinking maybe we could go to the store tomorrow.

Input: what is one plus three
Output: What is one plus three?

Input: can you uh write me an email to like john about the meeting on friday
Output: Can you write me an email to John about the meeting on Friday?

Input: eight times seven equals what
Output: Eight times seven equals what?

Input: explain to me how neural networks work
Output: Explain to me how neural networks work.

Input: um i'm not a person i'm just a computer program designed to answer questions
Output: I'm not a person, I'm just a computer program designed to answer questions.

Input: {transcript}
Output:"""


class OllamaFormatter:
    """Formats voice transcripts via a local Ollama instance."""

    def __init__(self, url: str, model: str, timeout: float = 15.0) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def format(self, raw_transcript: str, context_hint: str = "") -> str:
        """Send transcript to Ollama and return cleaned text.

        Uses /api/generate with a completion prompt so the model fills in a pattern
        rather than entering chat/answer mode. Falls back to raw_transcript on any error.
        """
        prompt = _COMPLETION_PROMPT_TEMPLATE.format(transcript=raw_transcript)
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 200},
        }
        try:
            resp = httpx.post(
                f"{self._url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            # Take only the first line — model must not continue the pattern
            result = raw.splitlines()[0].strip() if raw else ""
            return result or raw_transcript
        except Exception:
            log.exception("OllamaFormatter.format failed")
            return raw_transcript

    def is_available(self) -> bool:
        """Return True if the Ollama server responds at /api/tags."""
        try:
            resp = httpx.get(f"{self._url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Module-level helpers (kept for backward-compat with existing call sites)
# ---------------------------------------------------------------------------

_LEGACY_TIMEOUT = httpx.Timeout(10.0, connect=3.0)


@dataclass
class OllamaFormatterResult:
    """Output from the module-level format_text helper."""

    text: str
    latency_ms: float
    model: str
    success: bool


def format_text(raw: str, settings: Settings) -> OllamaFormatterResult:
    """Thin wrapper around OllamaFormatter for call sites that use Settings."""
    t0 = time.perf_counter()
    url = f"{settings.ollama_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": _COMPLETION_PROMPT_TEMPLATE.format(transcript=raw),
        "stream": False,
    }
    try:
        with httpx.Client(timeout=_LEGACY_TIMEOUT) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "").strip()
            elapsed = (time.perf_counter() - t0) * 1000
            return OllamaFormatterResult(
                text=text or raw,
                latency_ms=elapsed,
                model=settings.ollama_model,
                success=bool(text),
            )
    except Exception:
        elapsed = (time.perf_counter() - t0) * 1000
        log.exception("Ollama formatter failed")
        return OllamaFormatterResult(
            text=raw, latency_ms=elapsed, model=settings.ollama_model, success=False
        )


def is_ollama_available(base_url: str) -> bool:
    """Ping the Ollama server and return True if it responds."""
    try:
        with httpx.Client(timeout=httpx.Timeout(3.0)) as client:
            resp = client.get(f"{base_url}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False
