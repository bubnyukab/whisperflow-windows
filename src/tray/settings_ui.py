"""Tkinter settings window for WhisperFlow — three-tab layout."""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

from src.config.settings import Settings, save_settings

log = logging.getLogger(__name__)

_WHISPER_MODELS = ["tiny.en", "base.en", "medium.en", "large-v3"]
_LANGUAGES = ["auto", "en", "de", "fr", "es", "it", "pt", "zh", "ja", "ko", "ru"]
_BACKEND_OPTIONS = [
    ("fast",  "Fast only",              "< 5ms  — rule-based cleanup, always available"),
    ("local", "Local LLM (fine-tuned)", "~105ms GPU — fine-tuned Qwen2.5-1.5B"),
]

# Estimated latency strings shown in Advanced tab
_LATENCY_HINTS: dict[str, str] = {
    "tiny.en": "~50ms GPU / ~400ms CPU",
    "base.en": "~100ms GPU / ~700ms CPU",
    "medium.en": "~300ms GPU / ~3s CPU",
    "large-v3": "~600ms GPU / ~8s CPU",
}


class SettingsWindow:
    """Modal settings dialog with General / Formatter / Advanced tabs."""

    def __init__(self, settings: Settings, on_save: Callable[[Settings], None]) -> None:
        self._settings = settings
        self._on_save = on_save

    def run(self) -> None:
        """Create Tk root, build window, block on mainloop."""
        self._root = tk.Tk()
        self._root.withdraw()
        self._win = tk.Toplevel(self._root)
        self._win.title("WhisperFlow Settings")
        self._win.resizable(False, False)
        self._win.protocol("WM_DELETE_WINDOW", self._cancel)
        self._win.lift()
        self._win.focus_force()
        self._build_ui()
        self._root.mainloop()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self._win)
        notebook.pack(fill="both", expand=True, padx=12, pady=(12, 4))

        general_frame = ttk.Frame(notebook, padding=12)
        notebook.add(general_frame, text="General")
        self._build_general_tab(general_frame)

        formatter_frame = ttk.Frame(notebook, padding=12)
        notebook.add(formatter_frame, text="Formatter")
        self._build_formatter_tab(formatter_frame)

        advanced_frame = ttk.Frame(notebook, padding=12)
        notebook.add(advanced_frame, text="Advanced")
        self._build_advanced_tab(advanced_frame)

        status_frame = ttk.Frame(notebook, padding=12)
        notebook.add(status_frame, text="Status")
        self._build_status_tab(status_frame)

        self._status_var = tk.StringVar(master=self._root, value="")
        status_bar = ttk.Label(self._win, textvariable=self._status_var,
                               foreground="gray", padding=(12, 4))
        status_bar.pack(fill="x")

        btn_frame = ttk.Frame(self._win, padding=(12, 4, 12, 12))
        btn_frame.pack(fill="x")
        ttk.Button(btn_frame, text="Save", command=self._save).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Apply", command=self._apply).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Test mic", command=self._test_mic).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Cancel", command=self._cancel).pack(side="right", padx=4)

    def _build_general_tab(self, frame: ttk.Frame) -> None:
        pad = {"padx": 6, "pady": 4}

        ttk.Label(frame, text="Hotkey:").grid(row=0, column=0, sticky="w", **pad)
        self._hotkey_var = tk.StringVar(value=self._settings.hotkey)
        ttk.Entry(frame, textvariable=self._hotkey_var, state="readonly", width=20).grid(
            row=0, column=1, sticky="ew", **pad
        )
        self._record_btn = ttk.Button(frame, text="Record…", command=self._start_hotkey_capture)
        self._record_btn.grid(row=0, column=2, sticky="w", **pad)

        ttk.Label(frame, text="Recording mode:").grid(row=1, column=0, sticky="w", **pad)
        self._mode_var = tk.StringVar(value=self._settings.recording_mode)
        ttk.Radiobutton(frame, text="Hold", variable=self._mode_var, value="hold").grid(
            row=1, column=1, sticky="w", **pad
        )
        ttk.Radiobutton(frame, text="Toggle", variable=self._mode_var, value="toggle").grid(
            row=1, column=2, sticky="w", **pad
        )

        ttk.Label(frame, text="Language:").grid(row=2, column=0, sticky="w", **pad)
        self._lang_var = tk.StringVar(value=self._settings.language)
        ttk.Combobox(
            frame, textvariable=self._lang_var, values=_LANGUAGES,
            state="readonly", width=8,
        ).grid(row=2, column=1, sticky="w", **pad)

    def _build_formatter_tab(self, frame: ttk.Frame) -> None:
        pad = {"padx": 6, "pady": 4}

        self._backend_var = tk.StringVar(
            master=self._root, value=self._settings.formatter_backend
        )

        for i, (value, label, hint) in enumerate(_BACKEND_OPTIONS):
            base_row = i * 2
            ttk.Radiobutton(
                frame, text=label, variable=self._backend_var,
                value=value, command=self._refresh_formatter_sections,
            ).grid(row=base_row, column=0, columnspan=2, sticky="w", **pad)
            ttk.Label(frame, text=hint, foreground="gray").grid(
                row=base_row + 1, column=1, sticky="w", padx=(24, 6), pady=(0, 4)
            )

        # --- Local LLM section ---
        self._local_frame = ttk.LabelFrame(frame, text="Local LLM settings", padding=8)
        self._local_frame.grid(row=len(_BACKEND_OPTIONS) * 2, column=0, columnspan=2,
                               sticky="ew", padx=6, pady=(8, 4))

        ttk.Label(self._local_frame, text="Model path:").grid(
            row=0, column=0, sticky="nw", padx=4, pady=3
        )
        self._local_model_path_var = tk.StringVar(value=str(self._settings.local_model_path))
        self._model_name_var = tk.StringVar(master=self._root)
        self._model_dir_var  = tk.StringVar(master=self._root)
        path_cell = ttk.Frame(self._local_frame)
        path_cell.grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        ttk.Label(path_cell, textvariable=self._model_name_var).pack(anchor="w")
        ttk.Label(path_cell, textvariable=self._model_dir_var,
                  foreground="gray", wraplength=260, justify="left").pack(anchor="w")
        ttk.Button(
            self._local_frame, text="Browse...",
            command=self._browse_local_model,
        ).grid(row=1, column=0, sticky="w", padx=4, pady=3)
        ttk.Button(
            self._local_frame, text="Test model",
            command=self._test_local_model,
        ).grid(row=1, column=1, sticky="w", padx=4, pady=3)

        # --- LLM word threshold ---
        threshold_row = len(_BACKEND_OPTIONS) * 2 + 1
        ttk.Label(frame, text="LLM word threshold:").grid(
            row=threshold_row, column=0, sticky="w", **pad
        )
        self._threshold_var = tk.IntVar(value=self._settings.llm_word_threshold)
        ttk.Spinbox(
            frame, textvariable=self._threshold_var, from_=3, to=50, width=6,
        ).grid(row=threshold_row, column=1, sticky="w", **pad)
        ttk.Label(
            frame, text="Inputs shorter than this skip the LLM for speed",
            foreground="gray",
        ).grid(row=threshold_row + 1, column=0, columnspan=2, sticky="w", **pad)

        self._update_model_path_display(str(self._settings.local_model_path))
        self._refresh_formatter_sections()

    def _update_model_path_display(self, path_str: str) -> None:
        p = Path(path_str)
        self._model_name_var.set(p.name if p.name else path_str)
        self._model_dir_var.set(str(p.parent) if p.parent != p else "")

    def _build_advanced_tab(self, frame: ttk.Frame) -> None:
        pad = {"padx": 6, "pady": 4}

        ttk.Label(frame, text="Whisper model:").grid(row=0, column=0, sticky="w", **pad)
        self._whisper_var = tk.StringVar(value=self._settings.whisper_model)
        whisper_cb = ttk.Combobox(
            frame, textvariable=self._whisper_var, values=_WHISPER_MODELS,
            state="readonly", width=16,
        )
        whisper_cb.grid(row=0, column=1, sticky="w", **pad)
        self._whisper_var.trace_add("write", lambda *_: self._update_latency_label())

        ttk.Label(frame, text="VAD silence (ms):").grid(row=1, column=0, sticky="w", **pad)
        self._vad_var = tk.IntVar(value=self._settings.vad_silence_ms)
        ttk.Scale(
            frame, variable=self._vad_var, from_=200, to=1500,
            orient="horizontal", length=200,
            command=lambda v: (
                self._vad_var.set(int(float(v))),
                self._vad_hint_var.set(self._vad_hint(int(float(v)))),
            ),
        ).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Label(frame, textvariable=self._vad_var).grid(row=1, column=2, sticky="w", **pad)

        self._vad_hint_var = tk.StringVar(master=self._root, value="")
        ttk.Label(frame, textvariable=self._vad_hint_var, foreground="gray").grid(
            row=2, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4)
        )
        self._vad_hint_var.set(self._vad_hint(self._settings.vad_silence_ms))

        self._latency_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self._latency_var, foreground="gray").grid(
            row=3, column=0, columnspan=3, sticky="w", **pad
        )
        self._update_latency_label()

    def _build_status_tab(self, frame: ttk.Frame) -> None:
        pad = {"padx": 6, "pady": 6}

        llm_path = Path(self._settings.local_model_path)
        llm_text = (
            f"{llm_path.name} — file found" if llm_path.exists() else "model file not found"
        )

        self._status_whisper_var = tk.StringVar(
            master=self._root,
            value=f"{self._settings.whisper_model} — checking…",
        )
        self._status_llm_var = tk.StringVar(master=self._root, value=llm_text)
        self._status_hotkey_var = tk.StringVar(
            master=self._root,
            value=f"{self._settings.hotkey} ({self._settings.recording_mode} mode)",
        )

        rows = [
            ("Whisper model", self._status_whisper_var),
            ("Local LLM",     self._status_llm_var),
            ("Hotkey",        self._status_hotkey_var),
        ]
        for i, (label, var) in enumerate(rows):
            ttk.Label(frame, text=label, width=14, anchor="w").grid(
                row=i, column=0, sticky="w", **pad
            )
            ttk.Label(frame, textvariable=var, foreground="gray").grid(
                row=i, column=1, sticky="w", **pad
            )

        ttk.Button(frame, text="Refresh", command=self._refresh_status).grid(
            row=len(rows), column=0, columnspan=2, sticky="w", padx=6, pady=(12, 4)
        )

    # ------------------------------------------------------------------
    # Dynamic section visibility
    # ------------------------------------------------------------------

    def _refresh_status(self) -> None:
        """Re-read current UI values and update status tab StringVars on a thread."""
        whisper_model = self._whisper_var.get()
        llm_path_str = self._local_model_path_var.get()
        hotkey = self._hotkey_var.get()
        mode = self._mode_var.get()

        def _check() -> None:
            llm_path = Path(llm_path_str)
            llm_text = (
                f"{llm_path.name} — file found" if llm_path.exists() else "model file not found"
            )
            whisper_text = f"{whisper_model} — ready"
            hotkey_text = f"{hotkey} ({mode} mode)"
            # Catch TclError in case the window was closed before the thread finished.
            try:
                self._root.after(0, lambda: self._status_whisper_var.set(whisper_text))
                self._root.after(0, lambda: self._status_llm_var.set(llm_text))
                self._root.after(0, lambda: self._status_hotkey_var.set(hotkey_text))
            except Exception:
                pass

        threading.Thread(target=_check, daemon=True).start()

    def _refresh_formatter_sections(self) -> None:
        if self._backend_var.get() == "local":
            self._local_frame.grid()
        else:
            self._local_frame.grid_remove()

    @staticmethod
    def _vad_hint(ms: int) -> str:
        if ms < 300:
            return "⚠ May cut off fast speakers"
        elif ms <= 500:
            return "✓ Recommended for most users"
        elif ms <= 800:
            return "Tolerates longer mid-sentence pauses"
        else:
            return "Long pause before transcription triggers"

    def _update_latency_label(self) -> None:
        hint = _LATENCY_HINTS.get(self._whisper_var.get(), "")
        self._latency_var.set(f"Estimated Whisper latency: {hint}")

    # ------------------------------------------------------------------
    # Hotkey capture
    # ------------------------------------------------------------------

    def _start_hotkey_capture(self) -> None:
        """Switch Record… button into listening mode and bind key events."""
        self._record_btn.config(text="Press keys…")
        self._capturing_keys: set = set()
        self._win.bind("<KeyPress>", self._on_capture_key_press)
        self._win.bind("<KeyRelease>", self._on_capture_key_release)
        self._win.focus_set()

    _MODIFIER_KEYSYMS = {
        "Control_L", "Control_R", "Shift_L", "Shift_R",
        "Alt_L", "Alt_R", "Super_L", "Super_R",
    }

    _BLOCKED_FINAL_KEYS = {
        "return", "enter", "escape", "backspace", "tab",
        "delete", "space",
        "shift_l", "shift_r", "control_l", "control_r",
        "alt_l", "alt_r", "super_l", "super_r",
    }

    def _on_capture_key_press(self, event: tk.Event) -> None:
        if event.keysym in self._MODIFIER_KEYSYMS:
            self._capturing_keys.add(event.keysym)
            return
        if event.keysym.lower() in self._BLOCKED_FINAL_KEYS:
            self._status_var.set(
                f"Key '{event.keysym}' can't be used as a hotkey trigger — "
                "try a letter, number, or function key."
            )
            return
        parts: list[str] = []
        if event.state & 0x4:
            parts.append("ctrl")
        if event.state & 0x1:
            parts.append("shift")
        if event.state & 0x8:
            parts.append("alt")
        parts.append(event.keysym.lower())
        self._finish_hotkey_capture("+".join(parts))

    def _on_capture_key_release(self, event: tk.Event) -> None:
        self._capturing_keys.discard(event.keysym)

    def _finish_hotkey_capture(self, combo: str) -> None:
        self._hotkey_var.set(combo)
        self._win.unbind("<KeyPress>")
        self._win.unbind("<KeyRelease>")
        self._record_btn.config(text="Record…")

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _test_mic(self) -> None:
        self._status_var.set("Test mic: recording 3s…")
        threading.Thread(target=self._run_mic_test, daemon=True).start()

    def _run_mic_test(self) -> None:
        def _set(msg: str) -> None:
            # StringVar.set from a non-main thread is unsafe; schedule on main loop.
            # Catch TclError in case the window was closed before the thread finished.
            try:
                self._root.after(0, lambda: self._status_var.set(msg))
            except Exception:
                pass

        try:
            from src.transcription.whisper_engine import WhisperEngine
            import tempfile, os

            engine = WhisperEngine(self._settings.whisper_model)
            try:
                import sounddevice as sd
                import scipy.io.wavfile as wav

                sample_rate = 16000
                audio = sd.rec(3 * sample_rate, samplerate=sample_rate,
                               channels=1, dtype="int16")
                sd.wait()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp = f.name
                wav.write(tmp, sample_rate, audio)
                text = engine.transcribe(tmp)
                os.unlink(tmp)
                _set(f"Mic test: {text or '(no speech detected)'}")
            except ImportError:
                _set("sounddevice not installed — mic test unavailable")
        except Exception as exc:
            log.exception("Mic test failed")
            _set(f"Error: {exc}")

    def _test_local_model(self) -> None:
        self._status_var.set("Testing local model…")
        threading.Thread(target=self._run_local_model_test, daemon=True).start()

    def _run_local_model_test(self) -> None:
        def _set(msg: str) -> None:
            # Catch TclError in case the window was closed before the thread finished.
            try:
                self._root.after(0, lambda: self._status_var.set(msg))
            except Exception:
                pass

        path_str = self._local_model_path_var.get()
        if not Path(path_str).exists():
            _set(f"Model file not found: {path_str}")
            return
        try:
            from src.formatting.local_llm_formatter import LocalLLMFormatter
            formatter = LocalLLMFormatter(model_path=path_str)
            result = formatter.format("um testing the microphone right now")
            _set(f"Model OK — output: \"{result}\"")
        except Exception as exc:
            log.exception("Local model test failed")
            _set(f"Model failed to load: {exc}")

    def _browse_local_model(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select GGUF model file",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
        )
        if path:
            self._local_model_path_var.set(path)
            self._update_model_path_display(path)

    # ------------------------------------------------------------------
    # Save / apply / cancel
    # ------------------------------------------------------------------

    def _collect_settings(self) -> Settings:
        # All widget vars are unconditionally created in _build_ui(); no hasattr guards needed.
        return Settings(
            whisper_model=self._whisper_var.get(),
            hotkey=self._hotkey_var.get(),
            recording_mode=self._mode_var.get(),
            language=self._lang_var.get(),
            formatter_backend=self._backend_var.get(),
            llm_word_threshold=self._threshold_var.get(),
            vad_silence_ms=self._vad_var.get(),
            local_model_path=Path(self._local_model_path_var.get()),
            history_max=self._settings.history_max,
            models_dir=self._settings.models_dir,
            log_dir=self._settings.log_dir,
            training_pairs_path=self._settings.training_pairs_path,
        )

    def _save(self) -> None:
        updated = self._collect_settings()
        save_settings(updated)
        self._on_save(updated)
        if hasattr(self, "_root"):
            self._root.destroy()

    def _apply(self) -> None:
        updated = self._collect_settings()
        save_settings(updated)
        self._on_save(updated)
        self._settings = updated
        self._status_var.set("Settings applied.")

    def _cancel(self) -> None:
        if hasattr(self, "_root"):
            self._root.destroy()
