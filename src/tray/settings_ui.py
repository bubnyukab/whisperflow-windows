"""Tkinter settings window for WhisperFlow — three-tab layout."""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

from src.config.settings import Settings, check_ollama, save_settings

log = logging.getLogger(__name__)

_WHISPER_MODELS = ["tiny.en", "base.en", "medium.en", "large-v3"]
_OLLAMA_MODELS = ["phi3:mini", "mistral:7b", "llama3.1:8b", "gemma2:9b"]
_LANGUAGES = ["auto", "en", "de", "fr", "es", "it", "pt", "zh", "ja", "ko", "ru"]
_BACKENDS = ["Fast only", "Local LLM (fine-tuned)", "Ollama (local, free)"]

# Map display name → internal value
_BACKEND_VALUES = {
    "Fast only": "fast",
    "Local LLM (fine-tuned)": "local",
    "Ollama (local, free)": "ollama",
}
_BACKEND_LABELS = {v: k for k, v in _BACKEND_VALUES.items()}

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
        self._win.protocol("WM_DELETE_WINDOW", self._root.destroy)
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

        btn_frame = ttk.Frame(self._win, padding=(12, 4, 12, 12))
        btn_frame.pack(fill="x")
        ttk.Button(btn_frame, text="Save", command=self._save).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Test mic", command=self._test_mic).pack(side="left", padx=4)

    def _build_general_tab(self, frame: ttk.Frame) -> None:
        pad = {"padx": 6, "pady": 4}

        ttk.Label(frame, text="Hotkey:").grid(row=0, column=0, sticky="w", **pad)
        self._hotkey_var = tk.StringVar(value=self._settings.hotkey)
        ttk.Entry(frame, textvariable=self._hotkey_var, width=26).grid(
            row=0, column=1, columnspan=2, sticky="ew", **pad
        )

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

        ttk.Label(frame, text="Backend:").grid(row=0, column=0, sticky="w", **pad)
        initial_label = _BACKEND_LABELS.get(self._settings.formatter_backend, "Fast only")
        self._backend_var = tk.StringVar(value=initial_label)
        ttk.Combobox(
            frame, textvariable=self._backend_var, values=_BACKENDS,
            state="readonly", width=24,
        ).grid(row=0, column=1, sticky="ew", **pad)
        self._backend_var.trace_add("write", lambda *_: self._refresh_formatter_sections())

        # --- Ollama section ---
        self._ollama_frame = ttk.LabelFrame(frame, text="Ollama settings", padding=8)
        self._ollama_frame.grid(row=1, column=0, columnspan=2, sticky="ew",
                                padx=6, pady=(8, 4))

        ttk.Label(self._ollama_frame, text="URL:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self._ollama_url_var = tk.StringVar(value=self._settings.ollama_url)
        ttk.Entry(self._ollama_frame, textvariable=self._ollama_url_var, width=30).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=4, pady=3
        )

        ttk.Label(self._ollama_frame, text="Model:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self._ollama_model_var = tk.StringVar(value=self._settings.ollama_model)
        ttk.Combobox(
            self._ollama_frame, textvariable=self._ollama_model_var,
            values=_OLLAMA_MODELS, state="readonly", width=20,
        ).grid(row=1, column=1, sticky="w", padx=4, pady=3)

        self._ollama_status_var = tk.StringVar(value="")
        ttk.Label(self._ollama_frame, textvariable=self._ollama_status_var).grid(
            row=2, column=1, sticky="w", padx=4, pady=2
        )
        ttk.Button(
            self._ollama_frame, text="Check connection",
            command=self._check_ollama_connection,
        ).grid(row=2, column=0, sticky="w", padx=4, pady=3)

        # --- Local LLM section ---
        self._local_frame = ttk.LabelFrame(frame, text="Local LLM settings", padding=8)
        self._local_frame.grid(row=1, column=0, columnspan=2, sticky="ew",
                               padx=6, pady=(8, 4))

        ttk.Label(self._local_frame, text="Model path:").grid(
            row=0, column=0, sticky="w", padx=4, pady=3
        )
        self._local_model_path_var = tk.StringVar(value=str(self._settings.local_model_path))
        ttk.Label(
            self._local_frame, textvariable=self._local_model_path_var,
            wraplength=240, justify="left",
        ).grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(
            self._local_frame, text="Browse...",
            command=self._browse_local_model,
        ).grid(row=1, column=0, sticky="w", padx=4, pady=3)

        # --- LLM word threshold ---
        ttk.Label(frame, text="LLM word threshold:").grid(row=3, column=0, sticky="w", **pad)
        self._threshold_var = tk.IntVar(value=self._settings.llm_word_threshold)
        ttk.Spinbox(
            frame, textvariable=self._threshold_var, from_=3, to=50, width=6,
        ).grid(row=3, column=1, sticky="w", **pad)
        ttk.Label(
            frame, text="Inputs shorter than this skip the LLM for speed",
            foreground="gray",
        ).grid(row=4, column=0, columnspan=2, sticky="w", **pad)

        self._refresh_formatter_sections()

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
            command=lambda v: self._vad_var.set(int(float(v))),
        ).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Label(frame, textvariable=self._vad_var).grid(row=1, column=2, sticky="w", **pad)

        self._latency_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self._latency_var, foreground="gray").grid(
            row=2, column=0, columnspan=3, sticky="w", **pad
        )
        self._update_latency_label()

        # mic test output
        self._mic_result_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self._mic_result_var, wraplength=320,
                  justify="left").grid(row=3, column=0, columnspan=3, sticky="w", **pad)

    # ------------------------------------------------------------------
    # Dynamic section visibility
    # ------------------------------------------------------------------

    def _refresh_formatter_sections(self) -> None:
        backend_label = self._backend_var.get()
        backend = _BACKEND_VALUES.get(backend_label, "fast")
        if backend == "ollama":
            self._ollama_frame.grid()
            self._local_frame.grid_remove()
        elif backend == "local":
            self._ollama_frame.grid_remove()
            self._local_frame.grid()
        else:
            self._ollama_frame.grid_remove()
            self._local_frame.grid_remove()

    def _update_latency_label(self) -> None:
        model = self._whisper_var.get() if hasattr(self, "_whisper_var") else self._settings.whisper_model
        hint = _LATENCY_HINTS.get(model, "")
        self._latency_var.set(f"Estimated Whisper latency: {hint}")

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _check_ollama_connection(self) -> None:
        self._ollama_status_var.set("Checking...")

        def _check() -> None:
            import time
            url = self._ollama_url_var.get()
            t0 = time.perf_counter()
            ok = check_ollama(url)
            ms = int((time.perf_counter() - t0) * 1000)
            msg = f"OK ({ms}ms)" if ok else "FAIL — Ollama not reachable"
            self._root.after(0, lambda: self._ollama_status_var.set(msg))

        threading.Thread(target=_check, daemon=True).start()

    def _test_mic(self) -> None:
        if hasattr(self, "_mic_result_var"):
            self._mic_result_var.set("Recording 3s…")
        threading.Thread(target=self._run_mic_test, daemon=True).start()

    def _run_mic_test(self) -> None:
        try:
            import time
            from src.config.settings import load_settings
            from src.transcription.whisper_engine import WhisperEngine
            import tempfile, os

            # Use WhisperEngine in batch mode for the 3s clip
            engine = WhisperEngine(self._settings.whisper_model)
            # Record 3 seconds via sounddevice if available, else show message
            try:
                import sounddevice as sd
                import numpy as np
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
                if hasattr(self, "_mic_result_var"):
                    self._mic_result_var.set(text or "(no speech detected)")
            except ImportError:
                if hasattr(self, "_mic_result_var"):
                    self._mic_result_var.set("sounddevice not installed — mic test unavailable")
        except Exception as exc:
            log.exception("Mic test failed")
            if hasattr(self, "_mic_result_var"):
                self._mic_result_var.set(f"Error: {exc}")

    def _browse_local_model(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select GGUF model file",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
        )
        if path and hasattr(self, "_local_model_path_var"):
            self._local_model_path_var.set(path)

    # ------------------------------------------------------------------
    # Save / cancel
    # ------------------------------------------------------------------

    def _save(self) -> None:
        backend_label = self._backend_var.get() if hasattr(self, "_backend_var") else "Fast only"
        backend = _BACKEND_VALUES.get(backend_label, "fast")
        updated = Settings(
            whisper_model=self._whisper_var.get() if hasattr(self, "_whisper_var") else self._settings.whisper_model,
            hotkey=self._hotkey_var.get() if hasattr(self, "_hotkey_var") else self._settings.hotkey,
            recording_mode=self._mode_var.get() if hasattr(self, "_mode_var") else self._settings.recording_mode,
            language=self._lang_var.get() if hasattr(self, "_lang_var") else self._settings.language,
            formatter_backend=backend,
            ollama_model=self._ollama_model_var.get() if hasattr(self, "_ollama_model_var") else self._settings.ollama_model,
            ollama_url=self._ollama_url_var.get() if hasattr(self, "_ollama_url_var") else self._settings.ollama_url,
            ollama_timeout=self._settings.ollama_timeout,
            llm_word_threshold=self._threshold_var.get() if hasattr(self, "_threshold_var") else self._settings.llm_word_threshold,
            vad_silence_ms=self._vad_var.get() if hasattr(self, "_vad_var") else self._settings.vad_silence_ms,
            local_model_path=Path(self._local_model_path_var.get()) if hasattr(self, "_local_model_path_var") else self._settings.local_model_path,
            history_max=self._settings.history_max,
            models_dir=self._settings.models_dir,
            log_dir=self._settings.log_dir,
            training_pairs_path=self._settings.training_pairs_path,
        )
        save_settings(updated)
        self._on_save(updated)
        if hasattr(self, "_root"):
            self._root.destroy()
