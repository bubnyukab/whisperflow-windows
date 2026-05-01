"""Benchmark the Q4_K_M GGUF cleaner via llama-cpp-python with full GPU offload.

Loads a GGUF model, runs the same quality suite as evaluate_model.py (so we
can confirm Q4_K_M didn't degrade outputs), then times 20 generations on a
typical input with per-run completion-token counts.

Run from the project root:
    .venv/Scripts/python.exe tools/benchmark_gguf.py
    .venv/Scripts/python.exe tools/benchmark_gguf.py \
        --gguf models/whisperflow-cleaner-v2/model.gguf
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

# torch imported first so its bundled CUDA DLLs (cudart64_12.dll, cublas*,
# nvrtc*) are loaded into the process before llama_cpp probes for them.
import torch  # noqa: F401
from llama_cpp import Llama

from evaluate_model import (
    LATENCY_INPUT,
    LATENCY_RUNS,
    LATENCY_WARMUP,
    MAX_NEW_TOKENS,
    TEST_CASES,
    banner,
    looks_like_answer,
    render_case,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GGUF = ROOT / "models" / "whisperflow-cleaner" / "model.gguf"

# Stop on newline matches how the training data was framed: the cleaned
# output ends at the first \n before the next 'Input: ' would appear.
STOP_TOKENS = ["\n", "Input:"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a GGUF cleaner.")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF,
                        help="Path to GGUF model file")
    return parser.parse_args()


def clean_gguf(llm: Llama, text: str) -> tuple[str, int]:
    """Return (cleaned_text, n_completion_tokens)."""
    if text == "":
        return "", 0
    prompt = f"Input: {text}\nOutput: "
    out = llm.create_completion(
        prompt,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        stop=STOP_TOKENS,
        echo=False,
    )
    text_out = out["choices"][0]["text"].strip()
    n_tok = int(out["usage"]["completion_tokens"])
    return text_out, n_tok


def main() -> None:
    args = parse_args()
    banner("WhisperFlow cleaner - GGUF Q4_K_M benchmark (llama-cpp-python, CUDA)")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not args.gguf.exists():
        raise FileNotFoundError(f"Missing GGUF: {args.gguf}")
    print(f"Model: {args.gguf}  ({args.gguf.stat().st_size / 1024**3:.2f} GB)")

    print("Loading GGUF with full GPU offload (n_gpu_layers=-1) ...")
    llm = Llama(
        model_path=str(args.gguf),
        n_gpu_layers=-1,
        n_ctx=512,
        n_batch=512,
        seed=0,
        verbose=False,
    )

    banner("Quality test suite (post-Q4_K_M sanity)")
    flagged: list[int] = []
    for i, case in enumerate(TEST_CASES, 1):
        out, _ = clean_gguf(llm, case.text)
        f, reason = looks_like_answer(case, out)
        render_case(i, case, out, f, reason)
        if f:
            flagged.append(i)
    total = len(TEST_CASES)
    print(f"\n  {total - len(flagged)}/{total} clean, {len(flagged)} flagged")
    if flagged:
        print(f"  flagged cases: {', '.join(f'[{i:02d}]' for i in flagged)}")

    banner(f"Latency benchmark — {LATENCY_RUNS} runs (after {LATENCY_WARMUP} warmup)")
    print(f"input: {LATENCY_INPUT}")
    print(f"max_tokens cap: {MAX_NEW_TOKENS}\n")

    for _ in range(LATENCY_WARMUP):
        clean_gguf(llm, LATENCY_INPUT)

    timings_ms: list[float] = []
    token_counts: list[int] = []
    for i in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        _, n = clean_gguf(llm, LATENCY_INPUT)
        dt = (time.perf_counter() - t0) * 1000
        timings_ms.append(dt)
        token_counts.append(n)
        print(f"  run {i+1:02d}: {dt:7.1f} ms   tokens generated: {n}")

    avg_ms = statistics.mean(timings_ms)
    p50_ms = statistics.median(timings_ms)
    p95_ms = sorted(timings_ms)[int(0.95 * (LATENCY_RUNS - 1))]
    avg_tok = statistics.mean(token_counts)
    max_tok = max(token_counts)

    print(f"\n  avg : {avg_ms:7.1f} ms")
    print(f"  p50 : {p50_ms:7.1f} ms")
    print(f"  p95 : {p95_ms:7.1f} ms")
    print(f"  min : {min(timings_ms):7.1f} ms")
    print(f"  max : {max(timings_ms):7.1f} ms")
    print(f"  tokens: avg {avg_tok:.1f}, max {max_tok}, cap {MAX_NEW_TOKENS}")
    if avg_ms > 0:
        print(f"  throughput: {avg_tok / (avg_ms / 1000):.1f} tok/s")

    if max_tok >= MAX_NEW_TOKENS:
        print(f"\n  WARNING: at least one run hit max_tokens={MAX_NEW_TOKENS} — stop tokens not firing")
    else:
        print(f"\n  Stop firing cleanly (max generated = {max_tok}, cap = {MAX_NEW_TOKENS})")

    target_ms = 400
    if avg_ms < target_ms:
        print(f"  Target met: avg {avg_ms:.0f} ms < {target_ms} ms")
    else:
        print(f"  Target missed: avg {avg_ms:.0f} ms >= {target_ms} ms")


if __name__ == "__main__":
    main()
