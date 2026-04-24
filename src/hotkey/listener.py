"""Global hotkey listener using pynput (no root required on Linux)."""

from __future__ import annotations

import logging
import time
from typing import Callable

from pynput import keyboard

log = logging.getLogger(__name__)

_DEBOUNCE_S = 0.15
_KEY_ALIASES: dict[str, str] = {"win": "cmd", "windows": "cmd"}


def _to_pynput_format(hotkey: str) -> str:
    """Convert 'keyboard'-style 'ctrl+f9' to pynput '<ctrl>+<f9>'.

    Single alphanumeric characters stay bare; everything else gets angle brackets.
    'win' is aliased to 'cmd' (pynput's name for the Windows key).
    """
    tokens: list[str] = []
    for part in hotkey.lower().split("+"):
        part = part.strip()
        part = _KEY_ALIASES.get(part, part)
        tokens.append(part if (len(part) == 1 and part.isalnum()) else f"<{part}>")
    return "+".join(tokens)


class HotkeyListener:
    """Listens for a global hotkey and fires press/release callbacks.

    HOLD mode:   on_press fires when the combo is fully held, on_release fires
                 when any key in the combo is lifted.
    TOGGLE mode: on_press fires on 1st activation, on_release fires on 2nd.

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
        self._listener: keyboard.Listener | None = None
        self._hk: keyboard.HotKey | None = None
        self._hotkey_keys: frozenset = frozenset()
        self._last_press = 0.0
        self._last_release = 0.0
        self._toggled = False
        self._active = False  # True while combo is held (hold mode only)

    @property
    def is_listening(self) -> bool:
        return self._listening

    def start(self) -> None:
        """Register hotkey listener — non-blocking; runs in a daemon thread."""
        if self._listening:
            return
        pynput_hotkey = _to_pynput_format(self._hotkey)
        parsed = keyboard.HotKey.parse(pynput_hotkey)
        self._hotkey_keys = frozenset(parsed)
        self._hk = keyboard.HotKey(parsed, self._on_activate)
        self._active = False
        self._toggled = False
        self._listener = keyboard.Listener(
            on_press=self._raw_press,
            on_release=self._raw_release,
        )
        self._listener.start()
        self._listening = True
        log.debug("HotkeyListener started: %s (%s)", self._hotkey, self._mode)

    def stop(self) -> None:
        """Stop and unregister the hotkey listener."""
        self._listening = False
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                log.exception("HotkeyListener.stop failed to stop listener")
            self._listener = None
        self._hk = None
        log.debug("HotkeyListener stopped")

    # ------------------------------------------------------------------
    # pynput raw event handlers
    # ------------------------------------------------------------------

    def _raw_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if self._listener is None or self._hk is None:
            return
        try:
            self._hk.press(self._listener.canonical(key))
        except Exception:
            log.exception("HotkeyListener._raw_press error")

    def _raw_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if self._listener is None or self._hk is None:
            return
        try:
            canonical = self._listener.canonical(key)
            # Hold mode: firing any hotkey key while combo is active triggers release.
            if self._mode == "hold" and self._active and canonical in self._hotkey_keys:
                self._active = False
                self._handle_release()
            self._hk.release(canonical)
        except Exception:
            log.exception("HotkeyListener._raw_release error")

    def _on_activate(self) -> None:
        """Called by HotKey when all combo keys are simultaneously pressed."""
        if self._mode == "toggle":
            self._handle_toggle()
        else:
            self._active = True
            self._handle_press()

    # ------------------------------------------------------------------
    # Debounced callback dispatchers
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
