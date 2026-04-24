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

    if len(text.split()) < settings.llm_word_threshold:
        return fast_result

    if settings.formatter_backend == "fast":
        return fast_result

    llm_formatter = create_formatter(settings)
    result = llm_formatter.format(text, context_hint)
    return result if result else fast_result
