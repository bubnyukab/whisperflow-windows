"""Text formatting pipeline — fast rule-based and LLM-backed formatters."""

from __future__ import annotations

import logging

from src.config.settings import Settings
from src.formatting.claude_formatter import ClaudeFormatter
from src.formatting.fast_formatter import FastFormatter
from src.formatting.ollama_formatter import OllamaFormatter

log = logging.getLogger(__name__)


def create_formatter(settings: Settings) -> FastFormatter | OllamaFormatter | ClaudeFormatter:
    """Return the formatter instance for the configured backend."""
    backend = settings.formatter_backend
    if backend == "ollama":
        log.info("Formatter backend: ollama (%s @ %s)", settings.ollama_model, settings.ollama_url)
        return OllamaFormatter(settings.ollama_url, settings.ollama_model, settings.ollama_timeout)
    if backend == "claude":
        log.info("Formatter backend: claude")
        return ClaudeFormatter(settings.anthropic_api_key)
    log.info("Formatter backend: fast")
    return FastFormatter()


def format_text(text: str, settings: Settings, context_hint: str = "") -> str:
    """Two-tier formatter: always run FastFormatter, escalate to LLM when warranted."""
    fast_result = FastFormatter().format(text)
    word_count = len(text.split())

    log.debug("format_text: %d words, backend=%s, threshold=%d",
              word_count, settings.formatter_backend, settings.llm_word_threshold)

    if word_count < settings.llm_word_threshold:
        log.debug("format_text: using fast (below threshold)")
        return fast_result

    if settings.formatter_backend == "fast":
        log.debug("format_text: using fast (backend=fast)")
        return fast_result

    llm_formatter = create_formatter(settings)
    log.debug("format_text: calling LLM formatter")
    result = llm_formatter.format(text, context_hint)
    log.debug("format_text: LLM returned %r", result[:80] if result else None)
    return result if result else fast_result
