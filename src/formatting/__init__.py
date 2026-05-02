"""Text formatting pipeline — fast rule-based and LLM-backed formatters."""

from __future__ import annotations

import logging

from src.config.settings import Settings
from src.formatting.fast_formatter import FastFormatter

log = logging.getLogger(__name__)

# Module-level singletons — avoids re-constructing on every transcription.
_fast_formatter_instance = None

# LocalLLMFormatter loads a ~1 GB GGUF on construction, so we cache the
# instance across calls. Keyed on the model path so changing the path in
# settings invalidates the cache; an init failure is sticky-cached too so
# we don't retry the load on every keypress.
_local_instance = None
_local_cached_path: str | None = None
_local_init_failed: bool = False


def _get_fast_formatter() -> FastFormatter:
    """Return the module-level FastFormatter singleton."""
    global _fast_formatter_instance
    if _fast_formatter_instance is None:
        _fast_formatter_instance = FastFormatter()
    return _fast_formatter_instance


def reset_local_formatter() -> None:
    """Clear the cached LocalLLMFormatter so the next call reloads from disk."""
    global _local_instance, _local_cached_path, _local_init_failed
    _local_instance = None
    _local_cached_path = None
    _local_init_failed = False


def _get_local_formatter(settings: Settings):
    """Return a cached LocalLLMFormatter, or None if init failed."""
    global _local_instance, _local_cached_path, _local_init_failed
    from src.formatting.local_llm_formatter import LocalLLMFormatter

    path = str(settings.local_model_path)
    if path != _local_cached_path:
        _local_instance = None
        _local_init_failed = False
        _local_cached_path = path

    if _local_instance is None and not _local_init_failed:
        try:
            _local_instance = LocalLLMFormatter(path)
        except Exception:
            log.exception("LocalLLMFormatter init failed; will fall back to fast")
            _local_init_failed = True
    return _local_instance


def create_formatter(
    settings: Settings,
):
    """Return the formatter instance for the configured backend."""
    backend = settings.formatter_backend
    if backend == "local":
        local = _get_local_formatter(settings)
        if local is not None:
            log.info("Formatter backend: local (%s)", settings.local_model_path)
            return local
        log.warning("Formatter backend: local requested but unavailable; using fast")
        return _get_fast_formatter()
    log.info("Formatter backend: fast")
    return _get_fast_formatter()


def format_text(text: str, settings: Settings, context_hint: str = "") -> str:
    """Two-tier formatter: always run FastFormatter, escalate to LLM when warranted."""
    fast_result = _get_fast_formatter().format(text)
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
