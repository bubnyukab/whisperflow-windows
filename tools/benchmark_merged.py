"""Benchmark the merged fp16 cleaner: quality suite + latency with token counts.

Loads models/whisperflow-cleaner/merged-fp16/ as a vanilla HF model (no
PEFT, no bitsandbytes), runs the same test suite as evaluate_model.py to
verify quality survived merging, then times 20 generations on a typical
input and prints generated-token counts so we can confirm the model is
actually stopping at EOS rather than burning the full max_new_tokens cap.

Run from the project root:
    .venv/Scripts/python.exe tools/benchmark_merged.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_model import (
    LATENCY_INPUT,
    LATENCY_RUNS,
    LATENCY_WARMUP,
    MAX_NEW_TOKENS,
    TEST_CASES,
    banner,
    generate_with_tokens,
    looks_like_answer,
    render_case,
)

ROOT = Path(__file__).resolve().parent.parent
MERGED_DIR = ROOT / "models" / "whisperflow-cleaner" / "merged-fp16"


def main() -> None:
    banner("WhisperFlow cleaner — merged fp16 benchmark")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: not available — running on CPU (will be slow)")

    print(f"Loading merged fp16 model from {MERGED_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MERGED_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MERGED_DIR),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = True

    banner("Quality test suite (post-merge sanity)")
    flagged: list[int] = []
    for i, case in enumerate(TEST_CASES, 1):
        out, _ = generate_with_tokens(model, tokenizer, case.text)
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
    print(f"max_new_tokens cap: {MAX_NEW_TOKENS}\n")

    for _ in range(LATENCY_WARMUP):
        generate_with_tokens(model, tokenizer, LATENCY_INPUT)

    timings_ms: list[float] = []
    token_counts: list[int] = []
    for i in range(LATENCY_RUNS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, n = generate_with_tokens(model, tokenizer, LATENCY_INPUT)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
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
        print(f"\n  WARNING: at least one run hit max_new_tokens={MAX_NEW_TOKENS} — EOS not firing")
    else:
        print(f"\n  EOS firing cleanly (max generated = {max_tok}, cap = {MAX_NEW_TOKENS})")

    target_ms = 400
    if avg_ms < target_ms:
        print(f"  Target met: avg {avg_ms:.0f} ms < {target_ms} ms")
    else:
        print(f"  Target missed: avg {avg_ms:.0f} ms >= {target_ms} ms")


if __name__ == "__main__":
    main()
