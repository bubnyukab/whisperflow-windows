"""WhisperFlow Windows — entry point."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover
    from src.tray.tray_app import TrayApp
    from src.injection.text_injector import TextInjector

from src.config.settings import Settings, check_ollama, load_settings, SETTINGS_PATH
from src.formatting import format_text

log = logging.getLogger(__name__)

_HISTORY_PATH = Path.home() / ".whisperflow" / "history.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WhisperFlow Windows")
    parser.add_argument("--dev", action="store_true", help="Verbose debug logging")
    parser.add_argument("--model", metavar="MODEL",
                        help="Override Whisper model (tiny.en, base.en, …)")
    parser.add_argument(
        "--backend", choices=["fast", "ollama", "claude"],
        help="Override formatter backend",
    )
    return parser.parse_args()


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _apply_cli_overrides(args: argparse.Namespace, settings: Settings) -> Settings:
    """Return a new Settings with --model / --backend applied; original unchanged."""
    overrides: dict = {}
    if args.model:
        overrides["whisper_model"] = args.model
    if args.backend:
        overrides["formatter_backend"] = args.backend
    return replace(settings, **overrides) if overrides else settings


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def _append_history(entry: dict, path: Path, max_entries: int) -> None:
    """Append one entry to history.json, keeping only the last max_entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            if isinstance(raw, list):
                entries = raw
        except Exception:
            entries = []
    entries.append(entry)
    path.write_text(json.dumps(entries[-max_entries:], indent=2))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    raw_text: str,
    settings: Settings,
    tray: "TrayApp",
    injector: "TextInjector",
    history_path: Path,
) -> None:
    """Format → inject → history.  Never raises; surfaces errors via tray."""
    try:
        tray.set_state("processing")
        final_text = format_text(raw_text, settings)
        success = injector.inject(final_text)
        if not success:
            injector.show_fallback_toast(final_text)
        _append_history(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "raw_text": raw_text,
                "final_text": final_text,
            },
            history_path,
            settings.history_max,
        )
        tray.set_state("done")
    except Exception:
        log.exception("Pipeline error processing transcript")
        tray.show_notification("WhisperFlow Error", "Pipeline error — check logs.")


def _build_on_transcript(
    settings: Settings,
    tray: "TrayApp",
    injector: "TextInjector",
    history_path: Path,
) -> Callable[[str], None]:
    """Return a callback that runs _run_pipeline in a daemon thread."""
    def on_transcript(raw_text: str) -> None:
        threading.Thread(
            target=_run_pipeline,
            args=(raw_text, settings, tray, injector, history_path),
            daemon=True,
        ).start()
    return on_transcript


# ---------------------------------------------------------------------------
# Whisper pre-warm
# ---------------------------------------------------------------------------

def _prewarm_whisper(model: str) -> None:
    """Load the Whisper model weights in the background so first use is instant."""
    def _warm() -> None:
        try:
            from faster_whisper import WhisperModel
            WhisperModel(model, compute_type="auto")
            log.debug("Whisper model '%s' pre-warmed", model)
        except ImportError:
            log.debug("faster-whisper not installed — Whisper pre-warm skipped")
        except Exception:
            log.exception("Whisper pre-warm failed")
    threading.Thread(target=_warm, daemon=True).start()


# ---------------------------------------------------------------------------
# First-run wizard
# ---------------------------------------------------------------------------

def _run_first_run_wizard(settings: Settings) -> Settings:
    """Show a tkinter setup wizard on first launch; returns updated settings."""
    try:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title("WhisperFlow — First Run Setup")
        root.resizable(False, False)

        ollama_ok = check_ollama(settings.ollama_url)
        result: dict = {"settings": settings}

        frame = ttk.Frame(root, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Welcome to WhisperFlow!",
                  font=("Arial", 13, "bold")).pack(pady=(0, 10))

        if ollama_ok:
            ttk.Label(frame,
                      text="Ollama detected — local LLM formatting is available.",
                      foreground="green").pack()
            ttk.Label(frame, text="Ollama model:").pack(pady=(10, 0))
            model_var = tk.StringVar(value=settings.ollama_model)
            ttk.Combobox(frame, textvariable=model_var,
                         values=["phi3:mini", "mistral:7b", "llama3.1:8b", "gemma2:9b"],
                         state="readonly", width=20).pack()
            ttk.Label(frame,
                      text=f"\nHotkey: {settings.hotkey}\n"
                           "(change in Settings → General)",
                      justify="center").pack(pady=8)

            def _save_ollama() -> None:
                result["settings"] = replace(settings, ollama_model=model_var.get())
                root.destroy()

            ttk.Button(frame, text="Start WhisperFlow", command=_save_ollama).pack(pady=8)
        else:
            ttk.Label(frame, text="Ollama not detected.", foreground="orange").pack()
            ttk.Label(frame,
                      text="Install Ollama from https://ollama.com\n"
                           "then run:  ollama pull llama3.1:8b\n\n"
                           "Or continue in Fast-only mode (no LLM formatting).",
                      justify="center").pack(pady=8)

            def _fast_only() -> None:
                result["settings"] = replace(settings, formatter_backend="fast")
                root.destroy()

            ttk.Button(frame, text="Use Fast-only mode", command=_fast_only).pack(pady=4)
            ttk.Button(frame, text="I'll set up Ollama later",
                       command=root.destroy).pack()

        root.mainloop()
        return result["settings"]
    except Exception:
        log.warning("First-run wizard failed — using defaults", exc_info=True)
        return settings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Startup sequence → wire pipeline → block on tray."""
    from dotenv import load_dotenv
    load_dotenv()

    # 1. Settings
    config_existed = SETTINGS_PATH.exists()
    args = _parse_args()
    _setup_logging(args.dev)
    settings = load_settings()
    settings = _apply_cli_overrides(args, settings)

    # 2. First-run wizard
    if not config_existed:
        settings = _run_first_run_wizard(settings)

    # 3. Ollama health check
    ollama_down = (
        settings.formatter_backend == "ollama"
        and not check_ollama(settings.ollama_url)
    )
    if ollama_down:
        log.warning("Ollama unreachable at %s — will fall back to fast formatter",
                    settings.ollama_url)

    # 4. Pre-warm Whisper
    _prewarm_whisper(settings.whisper_model)

    # 5–7. Wire components
    from src.tray.tray_app import TrayApp
    from src.injection.text_injector import TextInjector
    from src.audio.realtime_recorder import RealtimeRecorder
    from src.hotkey.listener import HotkeyListener

    tray = TrayApp(settings)
    injector = TextInjector()

    on_transcript_cb = _build_on_transcript(settings, tray, injector, _HISTORY_PATH)
    recorder = RealtimeRecorder(settings, on_transcript=on_transcript_cb)

    def _on_press() -> None:
        tray.set_state("recording")
        recorder.start()

    def _on_release() -> None:
        recorder.stop()

    hotkey = HotkeyListener(
        hotkey=settings.hotkey,
        on_press=_on_press,
        on_release=_on_release,
        mode=settings.recording_mode,
    )
    hotkey.start()

    # Notify about Ollama after tray renders (small delay)
    if ollama_down:
        def _warn_ollama() -> None:
            import time
            time.sleep(0.8)
            tray.show_notification(
                "WhisperFlow",
                "Ollama is not running — using fast formatter.\n"
                "Start Ollama with: ollama serve",
            )
        threading.Thread(target=_warn_ollama, daemon=True).start()

    tray.run()  # blocks main thread


if __name__ == "__main__":
    main()
