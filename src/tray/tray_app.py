"""pystray-based system tray application — owns the main thread."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import pystray
from PIL import Image, ImageDraw

from src.audio.realtime_recorder import RealtimeRecorder
from src.config.settings import Settings
from src.formatting.fast_formatter import format_text as fast_format, word_count
from src.hotkey.listener import HotkeyListener
from src.injection.text_injector import inject

log = logging.getLogger(__name__)

_ICON_SIZE = (64, 64)


class TrayApp:
    """System tray application that orchestrates the full voice-to-text pipeline."""

    def __init__(self, settings: Settings) -> None:
        """Initialize all pipeline components.

        Args:
            settings: Loaded application settings.
        """
        self._settings = settings
        self._recorder = RealtimeRecorder(settings, on_text=self._on_transcript)
        self._hotkey = HotkeyListener(
            settings,
            on_press=self._recorder.start_session,
            on_release=self._recorder.end_session,
        )
        self._tray: Optional[pystray.Icon] = None
        self._check_ollama_async()

    def run(self) -> None:
        """Start the tray icon and block on the main thread."""
        menu = pystray.Menu(
            pystray.MenuItem("Settings", self._open_settings),
            pystray.MenuItem("Quit", self._quit),
        )
        self._tray = pystray.Icon("WhisperFlow", self._make_icon(), "WhisperFlow", menu)
        self._hotkey.start()
        self._tray.run()

    def notify(self, title: str, message: str) -> None:
        """Show a tray notification, logging if the tray is unavailable."""
        log.info("[notify] %s: %s", title, message)
        if self._tray:
            try:
                self._tray.notify(message, title)
            except Exception:
                pass

    def _on_transcript(self, text: str) -> None:
        """Pipeline entry — format and inject the transcript."""
        try:
            fast_result = fast_format(text)
            formatted = fast_result.text

            if (
                word_count(text) >= self._settings.llm_word_threshold
                and self._settings.formatter_backend != "fast"
            ):
                formatted = self._run_llm_formatter(fast_result.text)

            result = inject(formatted, self._settings)
            if not result.success:
                self.notify("WhisperFlow", f"Injection failed — text copied:\n{formatted}")
        except Exception:
            log.exception("Pipeline error in _on_transcript")
            self.notify("WhisperFlow Error", "Unexpected error — check logs.")

    def _run_llm_formatter(self, text: str) -> str:
        """Run the configured LLM formatter, falling back to the input on error."""
        backend = self._settings.formatter_backend
        try:
            if backend == "ollama":
                from src.formatting.ollama_formatter import format_text as ollama_fmt
                result = ollama_fmt(text, self._settings)
                if not result.success:
                    self.notify("WhisperFlow", "Ollama unavailable — used fast formatter.")
                return result.text
            if backend == "claude":
                from src.formatting.claude_formatter import format_text as claude_fmt
                result = claude_fmt(text, self._settings)
                if not result.success:
                    self.notify("WhisperFlow", "Claude API failed — used fast formatter.")
                return result.text
        except Exception:
            log.exception("LLM formatter (%s) raised unexpectedly", backend)
            self.notify("WhisperFlow", f"{backend} formatter failed — used fast formatter.")
        return text

    def _open_settings(self, icon: object, item: object) -> None:
        """Open the settings window in a daemon thread."""
        threading.Thread(target=self._show_settings_window, daemon=True).start()

    def _show_settings_window(self) -> None:
        """Instantiate and run the Tkinter settings window."""
        from src.tray.settings_ui import SettingsWindow
        SettingsWindow(self._settings, on_save=self._on_settings_saved).run()

    def _on_settings_saved(self, new_settings: Settings) -> None:
        """Reload settings and re-register the hotkey if it changed."""
        old_hotkey = self._settings.hotkey
        self._settings = new_settings
        self._recorder._settings = new_settings
        if new_settings.hotkey != old_hotkey:
            self._hotkey.stop()
            self._hotkey = HotkeyListener(
                new_settings,
                on_press=self._recorder.start_session,
                on_release=self._recorder.end_session,
            )
            self._hotkey.start()

    def _quit(self, icon: object, item: object) -> None:
        """Cleanly shut down the application."""
        self._hotkey.stop()
        self._recorder.shutdown()
        if self._tray:
            self._tray.stop()

    def _check_ollama_async(self) -> None:
        """Ping Ollama at startup in a background thread — warn if unreachable."""
        if self._settings.formatter_backend != "ollama":
            return

        def _check() -> None:
            from src.formatting.ollama_formatter import is_ollama_available
            if not is_ollama_available(self._settings.ollama_base_url):
                self.notify(
                    "WhisperFlow",
                    "Ollama is not running — falling back to fast formatter.\n"
                    "Start with: ollama serve",
                )

        threading.Thread(target=_check, daemon=True).start()

    @staticmethod
    def _make_icon() -> Image.Image:
        """Generate a simple microphone tray icon programmatically."""
        img = Image.new("RGBA", _ICON_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([4, 4, 60, 60], fill=(30, 144, 255))
        draw.ellipse([22, 16, 42, 44], fill="white")
        draw.rectangle([30, 44, 34, 56], fill="white")
        draw.rectangle([24, 56, 40, 60], fill="white")
        return img
