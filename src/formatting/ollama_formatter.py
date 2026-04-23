"""Ollama LLM formatter — calls local Ollama API to clean up transcripts."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from src.config.settings import Settings

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a transcription editor. Fix grammar, punctuation, and formatting of the "
    "spoken text below. Preserve the speaker's meaning exactly. "
    "Output only the corrected text, nothing else."
)

_TIMEOUT = httpx.Timeout(10.0, connect=3.0)


@dataclass
class OllamaFormatterResult:
    """Output from the Ollama formatter."""

    text: str
    latency_ms: float
    model: str
    success: bool


def format_text(raw: str, settings: Settings) -> OllamaFormatterResult:
    """Send raw transcript to the local Ollama API and return the formatted result.

    Falls back gracefully — callers should check result.success and fall back to
    the fast_formatter result when False.

    Args:
        raw: Raw or fast-formatted transcript.
        settings: App settings (ollama_base_url, ollama_model).
    """
    t0 = time.perf_counter()
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": f"{_SYSTEM_PROMPT}\n\n{raw}",
        "stream": False,
    }
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
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
