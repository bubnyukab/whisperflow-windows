"""Latency benchmark for WhisperFlow pipeline components."""

from __future__ import annotations

import argparse
import logging
import statistics
sys_path_patched = False
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys_path_patched = True
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)

_SAMPLE_TEXT = (
    "this is a test sentence that has more than ten words in it for the llm benchmark"
)


def run_fast_benchmark(runs: int) -> None:
    """Benchmark the fast formatter latency."""
    from src.formatting.fast_formatter import format_text

    print(f"\n--- FastFormatter benchmark ({runs} runs) ---")
    latencies: list[float] = []
    for _ in range(runs):
        result = format_text(_SAMPLE_TEXT)
        latencies.append(result.latency_ms)
    print(f"  median: {statistics.median(latencies)*1000:.2f}µs")
    print(f"  mean:   {statistics.mean(latencies)*1000:.2f}µs")
    print(f"  max:    {max(latencies)*1000:.2f}µs")


def run_whisper_benchmark(model: str, audio_path: Path, runs: int) -> None:
    """Benchmark the WhisperEngine transcription latency."""
    from pathlib import Path as _Path
    from src.transcription.whisper_engine import WhisperEngine

    print(f"\n--- Whisper benchmark ({model}, {runs} runs) ---")
    engine = WhisperEngine(model=model)
    latencies: list[float] = []
    for i in range(runs):
        result = engine.transcribe(_Path(audio_path))
        latencies.append(result.duration_ms)
        print(f"  run {i+1:2d}: {result.duration_ms:.1f}ms  text={result.text[:60]!r}")
    print(f"  median: {statistics.median(latencies):.1f}ms")
    print(f"  mean:   {statistics.mean(latencies):.1f}ms")
    print(f"  min:    {min(latencies):.1f}ms")
    print(f"  max:    {max(latencies):.1f}ms")


def run_ollama_benchmark(model: str, runs: int) -> None:
    """Benchmark the Ollama formatter latency."""
    from src.config.settings import Settings
    from src.formatting.ollama_formatter import format_text, is_ollama_available

    settings = Settings(ollama_model=model)
    if not is_ollama_available(settings.ollama_base_url):
        print("\n--- Ollama benchmark SKIPPED (server not running) ---")
        return

    print(f"\n--- Ollama benchmark ({model}, {runs} runs) ---")
    latencies: list[float] = []
    for i in range(runs):
        result = format_text(_SAMPLE_TEXT, settings)
        latencies.append(result.latency_ms)
        print(f"  run {i+1:2d}: {result.latency_ms:.1f}ms  ok={result.success}")
    print(f"  median: {statistics.median(latencies):.1f}ms")
    print(f"  mean:   {statistics.mean(latencies):.1f}ms")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="WhisperFlow latency benchmark")
    parser.add_argument("--model", default="tiny.en", help="Whisper model name")
    parser.add_argument(
        "--backend", default="all", choices=["all", "whisper", "ollama", "fast"]
    )
    parser.add_argument(
        "--audio", default="tests/fixtures/sample.wav", help="Audio file for Whisper benchmark"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--ollama-model", default="llama3.1:8b")
    args = parser.parse_args()

    from pathlib import Path
    audio_path = Path(args.audio)

    if args.backend in ("all", "fast"):
        run_fast_benchmark(args.runs)

    if args.backend in ("all", "whisper"):
        if not audio_path.exists():
            print(f"\n--- Whisper benchmark SKIPPED (no audio at {audio_path}) ---")
        else:
            run_whisper_benchmark(args.model, audio_path, args.runs)

    if args.backend in ("all", "ollama"):
        run_ollama_benchmark(args.ollama_model, args.runs)


if __name__ == "__main__":
    main()
