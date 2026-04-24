"""Global hotkey listener wrapping the keyboard library."""

from __future__ import annotations

import logging
import time
from typing import Callable

import keyboard

log = logging.getLogger(__name__)

_DEBOUNCE_S = 0.15


class HotkeyListener:
    """Listens for a global hotkey and fires press/release callbacks.

    HOLD mode:   on_press fires on keydown, on_release fires on keyup.
    TOGGLE mode: on_press fires on 1st keydown, on_release fires on 2nd keydown.

    Debounce: re-fires within 150 ms of the previous event are dropped.
    All callback exceptions are caught and logged — the listener never crashes.
    """

    def __init__(
        self,
        hotkey: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        mode: str = "hold",
    ) -> None:
        self._hotkey = hotkey
        self._on_press = on_press
        self._on_release = on_release
        self._mode = mode
        self._listening = False
        self._handles: list[object] = []
        self._last_press = 0.0
        self._last_release = 0.0
        self._toggled = False

    @property
    def is_listening(self) -> bool:
        return self._listening

    def start(self) -> None:
        """Register hotkeys — non-blocking; keyboard library runs its own thread."""
        if self._listening:
            return
        if self._mode == "toggle":
            h = keyboard.add_hotkey(self._hotkey, self._handle_toggle)
            self._handles = [h]
        else:  # hold
            h_down = keyboard.add_hotkey(self._hotkey, self._handle_press)
            h_up = keyboard.add_hotkey(
                self._hotkey, self._handle_release, trigger_on_release=True
            )
            self._handles = [h_down, h_up]
        self._listening = True
        log.debug("HotkeyListener started: %s (%s)", self._hotkey, self._mode)

    def stop(self) -> None:
        """Unregister all hotkey hooks."""
        self._listening = False
        for h in self._handles:
            try:
                keyboard.remove_hotkey(h)
            except Exception:
                log.exception("HotkeyListener.stop failed to remove hotkey")
        self._handles = []
        log.debug("HotkeyListener stopped")

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _handle_press(self) -> None:
        now = time.time()
        if now - self._last_press < _DEBOUNCE_S:
            return
        self._last_press = now
        try:
            self._on_press()
        except Exception:
            log.exception("HotkeyListener on_press callback raised")

    def _handle_release(self) -> None:
        now = time.time()
        if now - self._last_release < _DEBOUNCE_S:
            return
        self._last_release = now
        try:
            self._on_release()
        except Exception:
            log.exception("HotkeyListener on_release callback raised")

    def _handle_toggle(self) -> None:
        now = time.time()
        if now - self._last_press < _DEBOUNCE_S:
            return
        self._last_press = now
        if not self._toggled:
            self._toggled = True
            try:
                self._on_press()
            except Exception:
                log.exception("HotkeyListener on_press callback raised (toggle)")
        else:
            self._toggled = False
            try:
                self._on_release()
            except Exception:
                log.exception("HotkeyListener on_release callback raised (toggle)")
