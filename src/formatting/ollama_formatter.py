"""Ollama LLM formatter — calls local Ollama API to clean up transcripts."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

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


class OllamaFormatter:
    """Formats voice transcripts via a local Ollama instance."""

    def __init__(self, url: str, model: str, timeout: float = 15.0) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def format(self, raw_transcript: str, context_hint: str = "") -> str:
        """Send transcript to Ollama and return cleaned text.

        Falls back to raw_transcript on any error — never raises.
        """
        system = _SYSTEM_PROMPT
        if context_hint:
            system = f"{system}\n\nContext: {context_hint}"
        payload = {
            "model": self._model,
            "system": system,
            "prompt": raw_transcript,
            "stream": False,
        }
        try:
            resp = httpx.post(
                f"{self._url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip() or raw_transcript
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
        "prompt": f"{_SYSTEM_PROMPT}\n\n{raw}",
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
