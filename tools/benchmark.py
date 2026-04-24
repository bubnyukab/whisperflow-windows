"""Latency benchmark for the WhisperFlow pipeline.

Usage:
    python tools/benchmark.py [--model tiny.en] [--backend fast] [--iterations 5]

Runs tests/fixtures/sample.wav through each pipeline stage N times and prints
a min/avg/max table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Allow running directly from the project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

_STAGE_NAMES = [
    "Whisper transcribe",
    "Formatter (fast)",
    "Formatter (LLM)",
    "Text injection",
    "TOTAL",
]

_COL_STAGE = 21
_COL_NUM = 10


# ---------------------------------------------------------------------------
# Pure helpers (testable)
# ---------------------------------------------------------------------------

def _time_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Execute fn once and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - t0) * 1000


def _stats(timings: list[float]) -> tuple[float, float, float]:
    """Return (min, avg, max) for a list of ms timings."""
    return min(timings), sum(timings) / len(timings), max(timings)


def _format_table(rows: list[tuple[str, list[float] | None]]) -> str:
    """Format benchmark rows as the spec table.

    Each row is (stage_name, timings_list_or_None).
    None timings are rendered as N/A.
    """
    sep = "-" * _COL_STAGE + "+" + "-" * _COL_NUM + "+" + "-" * _COL_NUM + "+" + "-" * _COL_NUM
    header = (
        f"{'Stage':<{_COL_STAGE}}|"
        f"{'min (ms)':>{_COL_NUM}}|"
        f"{'avg (ms)':>{_COL_NUM}}|"
        f"{'max (ms)':>{_COL_NUM}}"
    )
    lines = [header, sep]
    for name, timings in rows:
        if timings is None:
            mn = av = mx = "N/A"
        else:
            lo, av_f, hi = _stats(timings)
            mn, av, mx = f"{lo:.1f}", f"{av_f:.1f}", f"{hi:.1f}"
        lines.append(
            f"{name:<{_COL_STAGE}}|{mn:>{_COL_NUM}}|{av:>{_COL_NUM}}|{mx:>{_COL_NUM}}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Thin wrappers — injected as seams so tests can mock them
# ---------------------------------------------------------------------------

def _transcribe(wav_path: Path, model: str) -> str:
    """Run WhisperEngine.transcribe; returns empty string if unavailable."""
    try:
        from src.transcription.whisper_engine import WhisperEngine
        return WhisperEngine(model=model).transcribe(str(wav_path))
    except ImportError:
        return ""
    except Exception as exc:
        print(f"  [Whisper] error: {exc}")
        return ""


def _inject(text: str) -> bool:
    """Run TextInjector.inject; returns False if unavailable."""
    try:
        from src.injection.text_injector import TextInjector
        return TextInjector().inject(text)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def run_benchmark(
    wav_path: Path,
    model: str,
    backend: str,
    iterations: int,
) -> list[tuple[str, list[float] | None]]:
    """Run the full pipeline N times and return per-stage timing rows."""
    from src.config.settings import Settings
    from src.formatting.fast_formatter import FastFormatter

    settings = Settings(whisper_model=model, formatter_backend=backend)
    fast_fmt = FastFormatter()

    whisper_timings: list[float] = []
    fast_timings: list[float] = []
    llm_timings: list[float] | None = [] if backend in ("ollama", "claude") else None
    inject_timings: list[float] = []

    # LLM availability check — skip if backend is fast or if server unreachable
    llm_available = False
    if backend == "ollama":
        from src.config.settings import check_ollama
        llm_available = check_ollama(settings.ollama_url)
        if not llm_available:
            print(f"  [Ollama] not reachable at {settings.ollama_url} — LLM stage skipped")
            llm_timings = None
    elif backend == "claude":
        llm_available = bool(settings.anthropic_api_key)
        if not llm_available:
            print("  [Claude] no API key — LLM stage skipped")
            llm_timings = None

    for i in range(iterations):
        # Stage 1 — Whisper
        transcript, w_ms = _time_ms(lambda: _transcribe(wav_path, model))
        whisper_timings.append(w_ms)

        text = transcript or "this is a benchmark test sentence with enough words to trigger llm"

        # Stage 2 — Fast formatter (always runs)
        _, f_ms = _time_ms(lambda: fast_fmt.format(text))
        fast_timings.append(f_ms)

        # Stage 3 — LLM formatter (optional)
        if llm_available and llm_timings is not None:
            if backend == "ollama":
                from src.formatting.ollama_formatter import OllamaFormatter
                llm = OllamaFormatter(settings.ollama_url, settings.ollama_model)
                _, l_ms = _time_ms(lambda: llm.format(text))
            else:
                from src.formatting.claude_formatter import ClaudeFormatter
                llm = ClaudeFormatter(settings.anthropic_api_key)
                _, l_ms = _time_ms(lambda: llm.format(text))
            llm_timings.append(l_ms)

        # Stage 4 — Text injection
        _, inj_ms = _time_ms(lambda: _inject(text))
        inject_timings.append(inj_ms)

        print(
            f"  run {i + 1}/{iterations}: "
            f"whisper={w_ms:.0f}ms  fast={f_ms:.2f}ms  inject={inj_ms:.0f}ms"
        )

    # Compute TOTAL per iteration
    total_timings: list[float] = []
    for idx in range(iterations):
        t = whisper_timings[idx] + fast_timings[idx] + inject_timings[idx]
        if llm_timings and idx < len(llm_timings):
            t += llm_timings[idx]
        total_timings.append(t)

    return [
        ("Whisper transcribe", whisper_timings),
        ("Formatter (fast)", fast_timings),
        ("Formatter (LLM)", llm_timings if (llm_timings is None or llm_timings) else None),
        ("Text injection", inject_timings),
        ("TOTAL", total_timings),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WhisperFlow latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="tiny.en",
                        help="Whisper model (tiny.en, base.en, medium.en, large-v3)")
    parser.add_argument("--backend", default="fast",
                        choices=["fast", "ollama", "claude"],
                        help="Formatter backend")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of benchmark iterations")
    parser.add_argument("--wav", default="tests/fixtures/sample.wav",
                        help="WAV file to use for transcription")
    return parser.parse_args()


def main() -> None:
    import logging
    logging.basicConfig(level=logging.WARNING)

    args = _parse_args()
    wav_path = Path(args.wav)

    if not wav_path.exists():
        print(f"WAV file not found: {wav_path}")
        sys.exit(1)

    print(f"\nWhisperFlow benchmark — model={args.model}, backend={args.backend}, "
          f"iterations={args.iterations}")
    print(f"Audio: {wav_path}\n")

    rows = run_benchmark(
        wav_path=wav_path,
        model=args.model,
        backend=args.backend,
        iterations=args.iterations,
    )

    print()
    print(_format_table(rows))
    print()


if __name__ == "__main__":
    main()
