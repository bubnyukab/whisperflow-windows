"""Global hotkey listener wrapping the keyboard library."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from src.config.settings import Settings

log = logging.getLogger(__name__)


class HotkeyListener:
    """Listens for a configurable global hotkey and fires press/release callbacks.

    Uses the keyboard library for Windows-compatible global hotkey detection.
    The press callback is guarded against repeated fires while the key is held.
    """

    def __init__(
        self,
        settings: Settings,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ) -> None:
        """Initialize the listener.

        Args:
            settings: App settings (hotkey string, e.g. 'win+shift+space').
            on_press: Called once when the hotkey combination is first pressed.
            on_release: Called when the trigger key is released.
        """
        self._settings = settings
        self._on_press = on_press
        self._on_release = on_release
        self._active = False

    def start(self) -> None:
        """Register global hotkey hooks."""
        import keyboard

        hotkey = self._settings.hotkey
        trigger_key = hotkey.split("+")[-1]
        try:
            keyboard.add_hotkey(hotkey, self._handle_press, suppress=False)
            keyboard.on_release_key(trigger_key, self._handle_release, suppress=False)
            log.info("Hotkey registered: %s", hotkey)
        except Exception:
            log.exception("Failed to register hotkey: %s", hotkey)

    def stop(self) -> None:
        """Unregister all hotkey hooks."""
        try:
            import keyboard
            keyboard.unhook_all()
        except Exception:
            log.exception("Failed to unhook keyboard hooks")

    def _handle_press(self) -> None:
        """Guard against repeated fires while the key is held."""
        if not self._active:
            self._active = True
            log.debug("Hotkey pressed")
            threading.Thread(target=self._on_press, daemon=True).start()

    def _handle_release(self, _event: object) -> None:
        """Fire the release callback once per press."""
        if self._active:
            self._active = False
            log.debug("Hotkey released")
            threading.Thread(target=self._on_release, daemon=True).start()
