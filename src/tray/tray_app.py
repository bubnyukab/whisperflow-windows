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


_TRAINING_COLOR = (255, 140, 0)  # orange indicator


def make_circle_icon(state: str, training_mode: bool = False) -> Image.Image:
    """Return a 64×64 RGBA image with a solid-color filled circle for state.

    When training_mode is True a small orange dot appears in the top-right corner.
    """
    color = _STATE_COLORS.get(state, _STATE_COLORS["idle"])
    img = Image.new("RGBA", _ICON_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, 60, 60], fill=(*color, 255))
    if training_mode:
        draw.ellipse([44, 4, 60, 20], fill=(*_TRAINING_COLOR, 255))
    return img


class TrayApp:
    """System tray application — owns the main thread via pystray.Icon.run()."""

    def __init__(
        self,
        settings: Settings,
        on_hotkey_changed: Optional[Callable[["Settings"], None]] = None,
    ) -> None:
        self._settings = settings
        self._on_hotkey_changed = on_hotkey_changed
        self._state = "idle"
        self._state_lock = threading.Lock()
        self._settings_lock = threading.Lock()  # guards _settings and _training_mode
        self._tray: Optional[pystray.Icon] = None
        self._done_timer: Optional[threading.Timer] = None
        self._indicator = RecordingIndicator()
        self._training_mode: bool = False
        # Window deduplication: tracks open singleton windows
        self._window_lock = threading.Lock()
        self._settings_window_open: bool = False
        self._history_window_open: bool = False
        self._training_pairs_window_open: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def training_mode(self) -> bool:
        """True when training mode is active."""
        with self._settings_lock:
            return self._training_mode

    def run(self) -> None:
        """Build the tray icon and block on the main thread."""
        menu = pystray.Menu(
            pystray.MenuItem("WhisperFlow", None, enabled=False),
            pystray.MenuItem(self._status_text, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings...", self._open_settings),
            pystray.MenuItem("View history", self._open_history),
            pystray.MenuItem(
                "Training Mode",
                self._toggle_training_mode,
                checked=lambda item: self._training_mode,
            ),
            pystray.MenuItem("View training pairs", self._open_training_pairs),
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
            training = self._training_mode

        if self._tray is not None:
            self._tray.icon = make_circle_icon(state, training)
            self._tray.update_menu()

        self._indicator.set_state(state)

        if state == "done":
            timer = threading.Timer(_DONE_RESET_DELAY, lambda: self.set_state("idle"))
            timer.daemon = True
            with self._state_lock:
                self._done_timer = timer
            timer.start()

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
        with self._window_lock:
            if self._settings_window_open:
                return
            self._settings_window_open = True
        threading.Thread(target=self._run_settings_window, daemon=True).start()

    def _run_settings_window(self) -> None:
        try:
            with self._settings_lock:
                settings = self._settings
            from src.tray.settings_ui import SettingsWindow
            SettingsWindow(settings, on_save=self._on_settings_saved).run()
        finally:
            with self._window_lock:
                self._settings_window_open = False

    def _on_settings_saved(self, new_settings: Settings) -> None:
        with self._settings_lock:
            old = self._settings
            self._settings = new_settings
        if self._on_hotkey_changed and (
            new_settings.hotkey != old.hotkey
            or new_settings.recording_mode != old.recording_mode
        ):
            self._on_hotkey_changed(new_settings)
        if (
            new_settings.formatter_backend != old.formatter_backend
            or new_settings.local_model_path != old.local_model_path
        ):
            from src.formatting import reset_local_formatter
            reset_local_formatter()
        if (
            new_settings.whisper_model != old.whisper_model
            or new_settings.vad_silence_ms != old.vad_silence_ms
        ):
            self.show_notification(
                "WhisperFlow",
                "Restart WhisperFlow to apply Whisper model / VAD changes.",
            )

    def _open_history(self, icon: object, item: object) -> None:
        with self._window_lock:
            if self._history_window_open:
                return
            self._history_window_open = True
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
        finally:
            with self._window_lock:
                self._history_window_open = False

    def _toggle_training_mode(self, icon: object, item: object) -> None:
        with self._settings_lock:
            self._training_mode = not self._training_mode
            training = self._training_mode
            state = self._state
        if self._tray is not None:
            self._tray.icon = make_circle_icon(state, training)
            self._tray.update_menu()
        log.info("Training mode %s", "ON" if training else "OFF")

    def _open_training_pairs(self, icon: object, item: object) -> None:
        with self._window_lock:
            if self._training_pairs_window_open:
                return
            self._training_pairs_window_open = True
        threading.Thread(target=self._show_training_pairs_window, daemon=True).start()

    def _show_training_pairs_window(self) -> None:
        try:
            self._do_show_training_pairs_window()
        finally:
            with self._window_lock:
                self._training_pairs_window_open = False

    def _do_show_training_pairs_window(self) -> None:
        from src.training.collector import TrainingCollector
        with self._settings_lock:
            settings = self._settings
        collector = TrainingCollector(settings.training_pairs_path)
        pairs = collector.load_pairs()

        try:
            import tkinter as tk
            from tkinter import ttk

            root = tk.Tk()
            root.title("WhisperFlow — Training Pairs")
            root.geometry("700x450")

            # Header
            header = tk.Frame(root, bg="#1e1e1e", pady=8)
            header.pack(fill="x")
            tk.Label(header, text=f"Training pairs collected: {len(pairs)}",
                     font=("Arial", 11, "bold"), bg="#1e1e1e", fg="white").pack(side="left",
                                                                                padx=12)

            def _export() -> None:
                import pyperclip
                pyperclip.copy(str(settings.training_pairs_path))
                tk.messagebox.showinfo("Exported",
                                       f"Path copied to clipboard:\n"
                                       f"{settings.training_pairs_path}")

            tk.Button(header, text="Export for training", command=_export,
                      bg="#2a4a6a", fg="white", relief="flat",
                      padx=10, pady=4).pack(side="right", padx=12)

            # Pairs list (last 10)
            text = tk.Text(root, wrap="word", state="disabled", padx=10, pady=8,
                           bg="#111111", fg="#dddddd", font=("Consolas", 9))
            scroll = ttk.Scrollbar(root, command=text.yview)
            text.configure(yscrollcommand=scroll.set)
            scroll.pack(side="right", fill="y")
            text.pack(fill="both", expand=True)

            text.configure(state="normal")
            if pairs:
                for entry in pairs[-10:]:
                    raw = entry.get("input", "")
                    cleaned = entry.get("output", "")
                    ts = entry.get("timestamp", "")
                    text.insert("end", f"[{ts}]\n")
                    text.insert("end", f"  Raw:   {raw}\n")
                    text.insert("end", f"  Clean: {cleaned}\n\n")
            else:
                text.insert("end", "No training pairs collected yet.\n\n"
                                   "Enable Training Mode from the tray menu, then dictate.")
            text.configure(state="disabled")

            root.mainloop()
        except Exception:
            log.warning("Training pairs window failed", exc_info=True)

    def _quit(self, icon: object, item: object) -> None:
        if self._tray:
            self._tray.stop()
