"""Floating recording-state pill overlay — shown during recording/processing/done."""

from __future__ import annotations

import logging
import threading
import tkinter as tk

log = logging.getLogger(__name__)

_BG = "#1a1a1a"
_W, _H = 220, 46
_PULSE_MS = 480   # dot blink cadence while recording

_STATE_CFG: dict[str, dict] = {
    "recording":  {"dot": "●", "dot_color": "#e03030", "text": "Recording…",  "text_color": "#ffffff", "pulse": True},
    "processing": {"dot": "◌", "dot_color": "#3a6eea", "text": "Processing…", "text_color": "#aaaaaa", "pulse": False},
    "done":       {"dot": "✓", "dot_color": "#33bb55", "text": "Done",         "text_color": "#ffffff", "pulse": False},
}


class RecordingIndicator:
    """Small always-on-top pill that mirrors WhisperFlow's recording state.

    Runs its own Tk mainloop on a daemon thread so it never blocks the caller.
    All public methods are thread-safe; they schedule updates via root.after().
    """

    def __init__(self) -> None:
        self._root: tk.Tk | None = None
        self._state = "idle"
        self._pulse_job: str | None = None
        self._dot_visible = True
        self._ready = threading.Event()

    def start(self) -> None:
        """Spawn the indicator thread and wait until the window is ready."""
        t = threading.Thread(target=self._run, daemon=True, name="recording-indicator")
        t.start()
        self._ready.wait(timeout=3)

    def set_state(self, state: str) -> None:
        """Schedule a state update on the indicator's own thread (safe to call from any thread)."""
        if self._root is not None:
            self._root.after(0, self._apply_state, state)

    # ------------------------------------------------------------------
    # Private — runs on the indicator thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._root = tk.Tk()
            self._root.overrideredirect(True)
            self._root.wm_attributes("-topmost", True)
            try:
                self._root.wm_attributes("-alpha", 0.93)
                self._root.wm_attributes("-toolwindow", True)  # keeps it off the taskbar
            except tk.TclError:
                pass  # platform doesn't support these attributes

            self._root.configure(bg=_BG)

            frame = tk.Frame(self._root, bg=_BG, padx=16, pady=11)
            frame.pack(fill="both", expand=True)

            self._dot_lbl = tk.Label(
                frame, text="●", font=("Segoe UI", 14),
                bg=_BG, fg="#e03030",
            )
            self._dot_lbl.pack(side="left", padx=(0, 9))

            self._text_lbl = tk.Label(
                frame, text="", font=("Segoe UI", 10),
                bg=_BG, fg="white",
            )
            self._text_lbl.pack(side="left")

            self._reposition()
            self._root.withdraw()  # hidden until first non-idle state
            self._ready.set()
            self._root.mainloop()
        except Exception:
            log.exception("RecordingIndicator thread crashed")
            self._ready.set()

    def _reposition(self) -> None:
        """Centre the pill horizontally, sit just above the taskbar."""
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        x = (sw - _W) // 2
        y = sh - _H - 72
        self._root.geometry(f"{_W}x{_H}+{x}+{y}")

    def _apply_state(self, state: str) -> None:
        self._state = state
        self._cancel_pulse()

        if state == "idle":
            self._root.withdraw()
            return

        cfg = _STATE_CFG.get(state)
        if cfg is None:
            self._root.withdraw()
            return

        self._dot_lbl.config(text=cfg["dot"], fg=cfg["dot_color"])
        self._text_lbl.config(text=cfg["text"], fg=cfg["text_color"])
        self._reposition()
        self._root.deiconify()
        self._root.lift()

        if cfg["pulse"]:
            self._dot_visible = True
            self._pulse()

    def _pulse(self) -> None:
        if self._state != "recording":
            return
        self._dot_visible = not self._dot_visible
        self._dot_lbl.config(fg="#e03030" if self._dot_visible else _BG)
        self._pulse_job = self._root.after(_PULSE_MS, self._pulse)

    def _cancel_pulse(self) -> None:
        if self._pulse_job is not None:
            self._root.after_cancel(self._pulse_job)
            self._pulse_job = None
