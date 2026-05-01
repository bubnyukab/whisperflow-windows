"""Tkinter popup for reviewing and correcting a transcription pair in training mode."""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk

from src.training.collector import TrainingCollector

log = logging.getLogger(__name__)

_AUTO_CLOSE_MS = 15_000
_WIN_W = 520
_WIN_H_COMPACT = 170
_WIN_H_EXPANDED = 290


class ReviewWindow:
    """Bottom-right popup that lets the user accept, edit, or skip a training pair.

    Call ``run()`` in a daemon thread — it blocks on its own tkinter mainloop.
    Injection happens immediately; this review is purely async.
    """

    def __init__(self, raw: str, cleaned: str, collector: TrainingCollector) -> None:
        self._raw = raw
        self._cleaned = cleaned
        self._collector = collector

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Build and show the review window; blocks until closed."""
        try:
            self._build_and_run()
        except Exception:
            log.warning("ReviewWindow failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_and_run(self) -> None:
        root = tk.Tk()
        root.title("WhisperFlow — Training Review")
        root.resizable(False, False)
        root.wm_attributes("-topmost", True)

        self._root = root
        self._expanded = False

        # Position bottom-right, above the normal toast area
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = sw - _WIN_W - 20
        y = sh - _WIN_H_COMPACT - 150  # 150px above taskbar / toast zone
        root.geometry(f"{_WIN_W}x{_WIN_H_COMPACT}+{x}+{y}")

        self._build_ui(root)

        # Auto-close: save as-is after 15 s
        self._timer_id = root.after(_AUTO_CLOSE_MS, self._accept)

        root.mainloop()

    def _build_ui(self, root: tk.Tk) -> None:
        pad = dict(padx=8, pady=4)

        frame = tk.Frame(root, bg="#1e1e1e")
        frame.pack(fill="both", expand=True)

        # ── Raw transcript row ──────────────────────────────────────────
        raw_frame = tk.Frame(frame, bg="#1e1e1e")
        raw_frame.pack(fill="x", **pad)
        tk.Label(raw_frame, text="Raw:", font=("Arial", 9, "bold"),
                 bg="#1e1e1e", fg="#888888", width=6, anchor="w").pack(side="left")
        tk.Label(raw_frame, text=self._raw[:120], font=("Arial", 9),
                 bg="#1e1e1e", fg="#cccccc", anchor="w", wraplength=430,
                 justify="left").pack(side="left", fill="x", expand=True)

        # ── Cleaned output row ──────────────────────────────────────────
        clean_frame = tk.Frame(frame, bg="#1e1e1e")
        clean_frame.pack(fill="x", **pad)
        tk.Label(clean_frame, text="Clean:", font=("Arial", 9, "bold"),
                 bg="#1e1e1e", fg="#888888", width=6, anchor="w").pack(side="left")
        tk.Label(clean_frame, text=self._cleaned[:120], font=("Arial", 9),
                 bg="#1e1e1e", fg="#ffffff", anchor="w", wraplength=430,
                 justify="left").pack(side="left", fill="x", expand=True)

        tk.Frame(frame, bg="#333333", height=1).pack(fill="x", padx=8, pady=2)

        # ── Countdown label ─────────────────────────────────────────────
        self._countdown_var = tk.StringVar(value="Auto-save in 15s")
        tk.Label(frame, textvariable=self._countdown_var, font=("Arial", 8),
                 bg="#1e1e1e", fg="#666666").pack(anchor="e", padx=8)
        self._remaining = _AUTO_CLOSE_MS // 1000
        self._tick()

        # ── Buttons ─────────────────────────────────────────────────────
        btn_frame = tk.Frame(frame, bg="#1e1e1e")
        btn_frame.pack(fill="x", padx=8, pady=4)

        tk.Button(btn_frame, text="✓ Looks good", command=self._accept,
                  bg="#2d6a2d", fg="white", relief="flat",
                  padx=10, pady=4).pack(side="left", padx=(0, 6))
        self._correct_btn = tk.Button(btn_frame, text="✎ Correct it",
                                      command=self._toggle_edit,
                                      bg="#4a4a1e", fg="white", relief="flat",
                                      padx=10, pady=4)
        self._correct_btn.pack(side="left", padx=(0, 6))
        tk.Button(btn_frame, text="✗ Skip", command=self._skip,
                  bg="#5a2020", fg="white", relief="flat",
                  padx=10, pady=4).pack(side="left")

        # ── Edit area (hidden until "Correct it") ───────────────────────
        self._edit_frame = tk.Frame(frame, bg="#1e1e1e")
        # not packed yet

        tk.Label(self._edit_frame, text="Edit correction:",
                 font=("Arial", 9, "bold"), bg="#1e1e1e", fg="#aaaaaa").pack(
            anchor="w", padx=4, pady=(4, 2))
        self._edit_box = tk.Text(self._edit_frame, height=4, font=("Arial", 9),
                                 wrap="word", padx=4, pady=4,
                                 bg="#2a2a2a", fg="white", insertbackground="white",
                                 relief="flat")
        self._edit_box.pack(fill="x", padx=4)
        self._edit_box.insert("1.0", self._cleaned)

        tk.Button(self._edit_frame, text="Save correction",
                  command=self._save_correction,
                  bg="#2a4a6a", fg="white", relief="flat",
                  padx=10, pady=4).pack(anchor="e", padx=4, pady=4)

    def _tick(self) -> None:
        """Update countdown label every second."""
        if self._remaining > 0:
            self._countdown_var.set(f"Auto-save in {self._remaining}s")
            self._remaining -= 1
            self._root.after(1000, self._tick)

    def _toggle_edit(self) -> None:
        if self._expanded:
            return
        self._expanded = True
        self._correct_btn.configure(state="disabled")
        self._edit_frame.pack(fill="x", padx=8, pady=(0, 4))
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        x = sw - _WIN_W - 20
        y = sh - _WIN_H_EXPANDED - 150
        self._root.geometry(f"{_WIN_W}x{_WIN_H_EXPANDED}+{x}+{y}")

    def _accept(self) -> None:
        """Save raw→cleaned pair as-is and close."""
        self._cancel_timer()
        self._collector.save_pair(self._raw, self._cleaned)
        self._root.destroy()

    def _save_correction(self) -> None:
        """Save raw→user-edited text and close."""
        self._cancel_timer()
        edited = self._edit_box.get("1.0", "end").strip()
        self._collector.save_pair(self._raw, edited if edited else self._cleaned)
        self._root.destroy()

    def _skip(self) -> None:
        """Discard this pair and close without saving."""
        self._cancel_timer()
        self._root.destroy()

    def _cancel_timer(self) -> None:
        if self._timer_id is not None:
            self._root.after_cancel(self._timer_id)
            self._timer_id = None
