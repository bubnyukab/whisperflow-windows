"""Tkinter-based settings window for WhisperFlow."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from src.config.settings import Settings


class SettingsWindow:
    """Modal settings dialog backed by tkinter."""

    def __init__(self, settings: Settings, on_save: Callable[[Settings], None]) -> None:
        """Build the settings window.

        Args:
            settings: Current application settings.
            on_save: Callback invoked with updated Settings when the user clicks Save.
        """
        self._settings = settings
        self._on_save = on_save
        self._root: tk.Tk | None = None

    def run(self) -> None:
        """Create and run the Tkinter event loop (blocks until window closes)."""
        self._root = tk.Tk()
        self._root.title("WhisperFlow Settings")
        self._root.resizable(False, False)
        self._build_ui()
        self._root.mainloop()

    def _build_ui(self) -> None:
        """Construct all UI widgets."""
        assert self._root is not None
        pad: dict = {"padx": 10, "pady": 5}

        frame = ttk.Frame(self._root, padding=16)
        frame.grid(sticky="nsew")

        ttk.Label(frame, text="Whisper model:").grid(row=0, column=0, sticky="w", **pad)
        self._whisper_var = tk.StringVar(value=self._settings.whisper_model)
        ttk.Combobox(
            frame,
            textvariable=self._whisper_var,
            values=["tiny.en", "base.en", "medium.en", "large-v3"],
            state="readonly",
            width=20,
        ).grid(row=0, column=1, sticky="ew", **pad)

        ttk.Label(frame, text="Hotkey:").grid(row=1, column=0, sticky="w", **pad)
        self._hotkey_var = tk.StringVar(value=self._settings.hotkey)
        ttk.Entry(frame, textvariable=self._hotkey_var, width=22).grid(
            row=1, column=1, sticky="ew", **pad
        )

        ttk.Label(frame, text="Formatter:").grid(row=2, column=0, sticky="w", **pad)
        self._backend_var = tk.StringVar(value=self._settings.formatter_backend)
        ttk.Combobox(
            frame,
            textvariable=self._backend_var,
            values=["fast", "ollama", "claude"],
            state="readonly",
            width=20,
        ).grid(row=2, column=1, sticky="ew", **pad)

        ttk.Label(frame, text="Ollama model:").grid(row=3, column=0, sticky="w", **pad)
        self._ollama_var = tk.StringVar(value=self._settings.ollama_model)
        ttk.Entry(frame, textvariable=self._ollama_var, width=22).grid(
            row=3, column=1, sticky="ew", **pad
        )

        ttk.Label(frame, text="LLM word threshold:").grid(row=4, column=0, sticky="w", **pad)
        self._threshold_var = tk.IntVar(value=self._settings.llm_word_threshold)
        ttk.Spinbox(
            frame, textvariable=self._threshold_var, from_=1, to=50, width=5
        ).grid(row=4, column=1, sticky="w", **pad)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(btn_frame, text="Save", command=self._save).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self._cancel).pack(side="left", padx=5)

    def _save(self) -> None:
        """Persist settings and invoke the save callback."""
        assert self._root is not None
        updated = Settings(
            whisper_model=self._whisper_var.get(),
            hotkey=self._hotkey_var.get(),
            formatter_backend=self._backend_var.get(),  # type: ignore[arg-type]
            ollama_model=self._ollama_var.get(),
            ollama_base_url=self._settings.ollama_base_url,
            llm_word_threshold=self._threshold_var.get(),
            inject_delay_ms=self._settings.inject_delay_ms,
        )
        updated.save()
        self._on_save(updated)
        self._root.destroy()

    def _cancel(self) -> None:
        """Close without saving."""
        assert self._root is not None
        self._root.destroy()
