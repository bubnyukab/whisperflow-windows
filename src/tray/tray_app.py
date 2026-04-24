"""pystray-based system tray application for WhisperFlow."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

import pystray
from PIL import Image, ImageDraw

from src.config.settings import Settings
from src.tray.recording_indicator import RecordingIndicator

log = logging.getLogger(__name__)

_ICON_SIZE = (64, 64)

_STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "idle": (150, 150, 150),
    "recording": (220, 50, 50),
    "processing": (50, 100, 220),
    "done": (50, 200, 50),
}

_STATE_LABELS: dict[str, str] = {
    "idle": "Ready",
    "recording": "Recording...",
    "processing": "Processing...",
    "done": "Done",
}

_DONE_RESET_DELAY = 1.5


def make_circle_icon(state: str) -> Image.Image:
    """Return a 64×64 RGBA image with a solid-color filled circle for state."""
    color = _STATE_COLORS.get(state, _STATE_COLORS["idle"])
    img = Image.new("RGBA", _ICON_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, 60, 60], fill=(*color, 255))
    return img


class TrayApp:
    """System tray application — owns the main thread via pystray.Icon.run()."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._state = "idle"
        self._state_lock = threading.Lock()
        self._tray: Optional[pystray.Icon] = None
        self._done_timer: Optional[threading.Timer] = None
        # Optional pipeline callbacks — wired by main.py (Prompt 11)
        self._on_hotkey_press: Optional[Callable] = None
        self._on_hotkey_release: Optional[Callable] = None
        self._indicator = RecordingIndicator()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Build the tray icon and block on the main thread."""
        menu = pystray.Menu(
            pystray.MenuItem("WhisperFlow", None, enabled=False),
            pystray.MenuItem(self._status_text, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings...", self._open_settings),
            pystray.MenuItem("View history", self._open_history),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )
        self._tray = pystray.Icon(
            "WhisperFlow",
            icon=make_circle_icon("idle"),
            title="WhisperFlow",
            menu=menu,
        )
        self._indicator.start()
        self._tray.run()

    def set_state(self, state: str) -> None:
        """Thread-safe state update — refreshes icon image and status label."""
        with self._state_lock:
            if self._done_timer is not None:
                self._done_timer.cancel()
                self._done_timer = None
            self._state = state

        if self._tray is not None:
            self._tray.icon = make_circle_icon(state)
            self._tray.update_menu()

        self._indicator.set_state(state)

        if state == "done":
            timer = threading.Timer(_DONE_RESET_DELAY, lambda: self.set_state("idle"))
            timer.daemon = True
            timer.start()
            with self._state_lock:
                self._done_timer = timer

    def show_notification(self, title: str, message: str) -> None:
        """Spawn a daemon thread that shows a tkinter toast (bottom-right, 4s)."""
        log.info("[notify] %s: %s", title, message)
        threading.Thread(
            target=self._show_toast,
            args=(title, message),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _status_text(self, item: object = None) -> str:
        """Dynamic status label shown in the tray menu."""
        return f"Status: {_STATE_LABELS.get(self._state, self._state)}"

    def _show_toast(self, title: str, message: str) -> None:
        """Borderless tkinter window, bottom-right corner, auto-closes after 4s."""
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()

            toast = tk.Toplevel(root)
            toast.wm_overrideredirect(True)
            toast.wm_attributes("-topmost", True)

            # Position bottom-right
            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()
            w, h = 320, 80
            x = sw - w - 20
            y = sh - h - 60
            toast.geometry(f"{w}x{h}+{x}+{y}")

            frame = tk.Frame(toast, bg="#222222", pady=8, padx=12)
            frame.pack(fill="both", expand=True)
            tk.Label(frame, text=title, font=("Arial", 10, "bold"),
                     bg="#222222", fg="white").pack(anchor="w")
            tk.Label(frame, text=message, font=("Arial", 9),
                     bg="#222222", fg="#cccccc", wraplength=290,
                     justify="left").pack(anchor="w")

            root.after(4000, root.destroy)
            root.mainloop()
        except Exception:
            log.warning("Toast notification failed", exc_info=True)

    def _open_settings(self, icon: object, item: object) -> None:
        threading.Thread(target=self._run_settings_window, daemon=True).start()

    def _run_settings_window(self) -> None:
        from src.tray.settings_ui import SettingsWindow
        SettingsWindow(self._settings, on_save=self._on_settings_saved).run()

    def _on_settings_saved(self, new_settings: Settings) -> None:
        self._settings = new_settings

    def _open_history(self, icon: object, item: object) -> None:
        threading.Thread(target=self._show_history_window, daemon=True).start()

    def _show_history_window(self) -> None:
        history_path = Path.home() / ".whisperflow" / "history.json"
        entries: list[dict] = []
        if history_path.exists():
            try:
                entries = json.loads(history_path.read_text())
            except Exception:
                log.warning("Could not read history", exc_info=True)

        try:
            import tkinter as tk
            from tkinter import ttk

            root = tk.Tk()
            root.title("WhisperFlow — History")
            root.geometry("600x400")

            text = tk.Text(root, wrap="word", state="disabled", padx=8, pady=8)
            scroll = ttk.Scrollbar(root, command=text.yview)
            text.configure(yscrollcommand=scroll.set)
            scroll.pack(side="right", fill="y")
            text.pack(fill="both", expand=True)

            text.configure(state="normal")
            if entries:
                for e in reversed(entries[-50:]):
                    ts = e.get("timestamp", "")
                    final = e.get("final_text", "")
                    text.insert("end", f"[{ts}]\n{final}\n\n")
            else:
                text.insert("end", "No history yet.")
            text.configure(state="disabled")

            root.mainloop()
        except Exception:
            log.warning("History window failed", exc_info=True)

    def _quit(self, icon: object, item: object) -> None:
        if self._tray:
            self._tray.stop()
