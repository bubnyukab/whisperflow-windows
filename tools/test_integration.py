"""Smoke test the integrated formatter pipeline using the local GGUF backend.

Drives src.formatting.format_text() with formatter_backend='local' on the same
12-case suite used during evaluation, so we can confirm the model loads via
the cached LocalLLMFormatter and produces the expected outputs end-to-end.

Run from the project root:
    .venv/Scripts/python.exe tools/test_integration.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

# Allow `from src...` when run as `python tools/test_integration.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import load_settings  # noqa: E402
from src.formatting import format_text  # noqa: E402

from evaluate_model import TEST_CASES, banner, looks_like_answer, render_case  # noqa: E402


def main() -> None:
    banner("Integration smoke test — formatter_backend='local'")

    base = load_settings()
    # Force the local backend regardless of saved config; threshold=0 routes
    # every case (including single-word edges) through the LLM rather than
    # the fast path, so we exercise the integrated dispatch end-to-end.
    settings = replace(base, formatter_backend="local", llm_word_threshold=0)
    print(f"Model path: {settings.local_model_path}")
    print(f"Word threshold (overridden): {settings.llm_word_threshold}\n")

    flagged: list[int] = []
    timings_ms: list[float] = []
    for i, case in enumerate(TEST_CASES, 1):
        t0 = time.perf_counter()
        out = format_text(case.text, settings)
        dt = (time.perf_counter() - t0) * 1000
        timings_ms.append(dt)

        f, reason = looks_like_answer(case, out)
        render_case(i, case, out, f, reason)
        print(f"     latency: {dt:.1f} ms")
        if f:
            flagged.append(i)

    banner("Summary")
    total = len(TEST_CASES)
    print(f"  {total - len(flagged)}/{total} clean, {len(flagged)} flagged")
    if flagged:
        print(f"  flagged: {', '.join(f'[{i:02d}]' for i in flagged)}")

    nonempty = [t for t, c in zip(timings_ms, TEST_CASES) if c.text]
    if nonempty:
        print(f"  per-call latency over {len(nonempty)} non-empty cases:")
        print(f"    first call (incl. model load): {timings_ms[0]:.1f} ms")
        print(f"    avg of remaining             : {sum(nonempty[1:]) / max(1, len(nonempty) - 1):.1f} ms")


if __name__ == "__main__":
    main()
