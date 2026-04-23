"""Clipboard-based text injector for Windows."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import pyperclip

from src.config.settings import Settings

log = logging.getLogger(__name__)


@dataclass
class InjectionResult:
    """Result of a text injection attempt."""

    success: bool
    text: str
    error: Optional[str] = None


def inject(text: str, settings: Settings) -> InjectionResult:
    """Inject text into the focused application via clipboard paste.

    Saves the existing clipboard content, sets the new text, simulates Ctrl+V,
    then restores the previous clipboard after inject_delay_ms.

    Args:
        text: The text to inject.
        settings: App settings (inject_delay_ms).
    """
    try:
        previous = _safe_get_clipboard()
        pyperclip.copy(text)
        _simulate_paste()
        delay_s = settings.inject_delay_ms / 1000.0
        threading.Timer(delay_s, lambda: _safe_set_clipboard(previous)).start()
        return InjectionResult(success=True, text=text)
    except Exception as exc:
        log.exception("Text injection failed")
        return InjectionResult(success=False, text=text, error=str(exc))


def _safe_get_clipboard() -> str:
    """Return current clipboard text, or empty string if clipboard holds non-text."""
    try:
        return pyperclip.paste() or ""
    except Exception:
        return ""


def _safe_set_clipboard(text: str) -> None:
    """Restore clipboard to a previous value, ignoring errors."""
    try:
        pyperclip.copy(text)
    except Exception:
        pass


def _simulate_paste() -> None:
    """Send Ctrl+V to the focused window."""
    import keyboard
    keyboard.send("ctrl+v")
