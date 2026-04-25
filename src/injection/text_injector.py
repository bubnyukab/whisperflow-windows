"""Clipboard-based text injector for WhisperFlow."""

from __future__ import annotations

import logging
import time

import pyperclip
from pynput.keyboard import Controller as _KbController, Key as _Key

log = logging.getLogger(__name__)

_TOAST_TIMEOUT_MS = 15_000
_TOAST_WIDTH = 420
_TOAST_HEIGHT = 180


def _send_ctrl_v() -> None:
    """Simulate a Ctrl+V keystroke via pynput."""
    _kb = _KbController()
    with _kb.pressed(_Key.ctrl):
        _kb.press('v')
        _kb.release('v')


class TextInjector:
    """Injects text into the focused application via clipboard paste."""

    def inject(self, text: str) -> bool:
        """Copy text to clipboard, send Ctrl+V, then restore old clipboard.

        Returns True on success, False on any error.
        """
        try:
            # 1. Save old clipboard (may be non-text)
            old_clipboard: str | None = None
            try:
                old_clipboard = pyperclip.paste()
            except Exception:
                pass  # non-text clipboard — skip restore

            # 2. Set new text
            pyperclip.copy(text)

            # 3. Brief settle before paste
            time.sleep(0.05)

            # 4. Simulate paste
            _send_ctrl_v()

            # 5. Wait for application to consume paste before restoring
            time.sleep(0.5)

            # 6. Restore if old clipboard was text
            if old_clipboard is not None:
                try:
                    pyperclip.copy(old_clipboard)
                except Exception:
                    pass

            return True
        except Exception:
            log.exception("TextInjector.inject failed")
            return False

    def show_fallback_toast(self, text: str) -> None:
        """Show a borderless always-on-top window with the transcript and a Copy button.

        Auto-closes after 15 seconds. Never raises — failure is silently logged.
        """
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()

            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()

            top = tk.Toplevel(root)
            top.overrideredirect(True)
            top.wm_attributes("-topmost", True)

            pad = 16
            x = sw - _TOAST_WIDTH - pad
            y = sh - _TOAST_HEIGHT - pad
            top.geometry(f"{_TOAST_WIDTH}x{_TOAST_HEIGHT}+{x}+{y}")

            # Scrollable text area
            frame = tk.Frame(top, bd=1, relief="solid")
            frame.pack(fill="both", expand=True, padx=8, pady=(8, 4))

            scrollbar = tk.Scrollbar(frame)
            scrollbar.pack(side="right", fill="y")

            txt = tk.Text(frame, wrap="word", yscrollcommand=scrollbar.set,
                          height=4, font=("Segoe UI", 10))
            txt.insert("1.0", text)
            txt.config(state="disabled")
            txt.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=txt.yview)

            # Copy button
            def _copy() -> None:
                pyperclip.copy(text)

            btn = tk.Button(top, text="Copy", command=_copy,
                            font=("Segoe UI", 9), padx=12)
            btn.pack(pady=(0, 8))

            # Auto-close after 15 s
            root.after(_TOAST_TIMEOUT_MS, root.destroy)

            root.mainloop()
        except Exception:
            log.exception("TextInjector.show_fallback_toast failed")
