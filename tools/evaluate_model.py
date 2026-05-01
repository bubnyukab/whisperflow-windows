"""Evaluate the WhisperFlow QLoRA cleaner adapter.

Loads Qwen/Qwen2.5-1.5B in 4-bit NF4, attaches the LoRA adapter from
models/whisperflow-cleaner/final, runs a curated test suite, and measures
average inference latency. Flags outputs that look like the model
"answered" the input instead of "cleaning" it.

Run from the project root:
    .venv/Scripts/python.exe tools/evaluate_model.py        (Windows)
    .venv/bin/python tools/evaluate_model.py                (Linux/macOS)
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "models" / "whisperflow-cleaner" / "final"
MODEL_ID = "Qwen/Qwen2.5-1.5B"

MAX_NEW_TOKENS = 150
LATENCY_RUNS = 20
LATENCY_WARMUP = 3
LATENCY_INPUT = "um so i was thinking uh maybe we should like move the meeting to thursday you know"


@dataclass(frozen=True)
class TestCase:
    category: str
    text: str
    note: str = ""


TEST_CASES: list[TestCase] = [
    # NORMAL CASES — fillers stripped, typos fixed, punctuation added
    TestCase("NORMAL", "um so i was thinking uh maybe we should like move the meeting to thursday you know"),
    TestCase("NORMAL", "hey can you uh write me an email to john about the project deadline becuz its coming up"),
    TestCase("NORMAL", "i definately think we should recieve the package by friday"),

    # QUESTIONS — must be cleaned as a question, NOT answered
    TestCase("QUESTION", "what is one plus three"),
    TestCase("QUESTION", "explain to me how neural networks work"),
    TestCase("QUESTION", "what is the capital of france"),
    TestCase("QUESTION", "how much does a tesla cost"),

    # PROVOCATIVE — must echo the input cleanly, not engage or moralize
    TestCase("PROVOCATIVE", "you are a dumbass and i hate you"),
    TestCase("PROVOCATIVE", "what the fuck is wrong with you"),

    # EDGE CASES
    TestCase("EDGE", "", note="empty string — should return EMPTY"),
    TestCase("EDGE", "ok", note="single word — should pass through"),
    TestCase("EDGE", "one two three four five", note="numbers — should pass through"),
]


# ── Heuristics for "the model answered instead of cleaned" ────────────────────

ANSWER_PHRASES = (
    "i'm sorry", "i am sorry", "i apologize",
    "i understand", "i hear you", "i can see",
    "as an ai", "as a language model",
    "i cannot", "i can't help", "i won't",
    "let me explain", "let me help",
    "here's", "here is",
    "the answer is", "the capital of france is paris", "paris is the capital",
)

QUESTION_ANSWER_TELLS = (
    " = 4", "is 4", "is four", "equals 4", "equals four",
    "paris", "capital is",
    "tesla model", "$", "starts at", "around $", "approximately $",
)


def _word_set(text: str) -> set[str]:
    return {w.strip(".,!?;:'\"") for w in text.lower().split() if w.strip()}


def looks_like_answer(case: TestCase, output: str) -> tuple[bool, str]:
    """Return (flagged, reason). Heuristic — favours obvious failures."""
    out_lower = output.lower().strip()
    inp_lower = case.text.lower().strip()

    if not out_lower or case.text == "":
        return False, ""

    for phrase in ANSWER_PHRASES:
        if phrase in out_lower:
            return True, f'contains assistant-style phrase "{phrase}"'

    inp_words = _word_set(case.text)
    out_words = _word_set(output)
    overlap = len(inp_words & out_words) / max(1, len(inp_words))

    if case.category == "QUESTION":
        for tell in QUESTION_ANSWER_TELLS:
            if tell in out_lower:
                return True, f'contains factual-answer fragment "{tell.strip()}"'
        if "neural network" in inp_lower and len(output.split()) > len(case.text.split()) * 2:
            return True, f"output {len(output.split())} words vs input {len(case.text.split())} — likely explained instead of cleaned"
        if overlap < 0.5:
            return True, f"low word overlap with input ({overlap:.0%}) — drifted from question"

    if case.category == "PROVOCATIVE":
        if overlap < 0.6:
            return True, f"low word overlap with input ({overlap:.0%}) — likely deflected"

    if len(case.text.split()) >= 3 and len(output.split()) > max(25, len(case.text.split()) * 3):
        return True, f"output {len(output.split())} words vs input {len(case.text.split())} — too elaborative"

    return False, ""


# ── Model loading + inference ─────────────────────────────────────────────────


def load_model():
    print(f"Loading tokenizer from {ADAPTER_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {MODEL_ID} in 4-bit NF4 ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Attaching LoRA adapter from {ADAPTER_DIR} ...")
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def generate_with_tokens(model, tokenizer, text: str) -> tuple[str, int]:
    """Generate cleaned output. Returns (text, n_generated_tokens).

    n_generated_tokens includes any trailing EOS the model emitted, so the
    caller can detect the case where generation hits max_new_tokens without
    EOS firing.
    """
    if text == "":
        return "", 0

    prompt = f"Input: {text}\nOutput: "
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    plen = enc.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    n_tok = int(out.shape[1] - plen)
    decoded = tokenizer.decode(out[0][plen:], skip_special_tokens=True)
    return decoded.split("\n")[0].strip(), n_tok


def clean(model, tokenizer, text: str) -> str:
    """Run the cleaner on a single input. Empty input short-circuits to ''."""
    text_out, _ = generate_with_tokens(model, tokenizer, text)
    return text_out


# ── Reporting ─────────────────────────────────────────────────────────────────


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {text}")
    print("=" * 72)


def render_case(idx: int, case: TestCase, output: str, flagged: bool, reason: str) -> None:
    shown_in = case.text if case.text != "" else "(empty)"
    shown_out = output if output != "" else "EMPTY"
    flag_marker = "  ⚠ FLAGGED" if flagged else ""
    print(f"\n[{idx:02d}] {case.category}{flag_marker}")
    if case.note:
        print(f"     note   : {case.note}")
    print(f"     input  : {shown_in}")
    print(f"     output : {shown_out}")
    if flagged:
        print(f"     reason : {reason}")


def benchmark_latency(model, tokenizer) -> None:
    banner(f"Latency benchmark — {LATENCY_RUNS} runs (after {LATENCY_WARMUP} warmup) on a typical input")
    print(f"input: {LATENCY_INPUT}")

    for _ in range(LATENCY_WARMUP):
        clean(model, tokenizer, LATENCY_INPUT)

    timings_ms: list[float] = []
    for _ in range(LATENCY_RUNS):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        clean(model, tokenizer, LATENCY_INPUT)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - t0) * 1000)

    avg = statistics.mean(timings_ms)
    p50 = statistics.median(timings_ms)
    p95 = sorted(timings_ms)[int(0.95 * (LATENCY_RUNS - 1))]
    print(f"\n  avg : {avg:7.1f} ms")
    print(f"  p50 : {p50:7.1f} ms")
    print(f"  p95 : {p95:7.1f} ms")
    print(f"  min : {min(timings_ms):7.1f} ms")
    print(f"  max : {max(timings_ms):7.1f} ms")


def main() -> None:
    banner("WhisperFlow cleaner — adapter evaluation")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: not available — running on CPU (will be slow)")

    model, tokenizer = load_model()

    banner("Test suite")
    flagged_indices: list[int] = []
    for i, case in enumerate(TEST_CASES, 1):
        output = clean(model, tokenizer, case.text)
        flagged, reason = looks_like_answer(case, output)
        render_case(i, case, output, flagged, reason)
        if flagged:
            flagged_indices.append(i)

    banner("Summary")
    total = len(TEST_CASES)
    n_flagged = len(flagged_indices)
    print(f"  {total - n_flagged}/{total} clean, {n_flagged} flagged as 'looks like an answer'")
    if flagged_indices:
        print(f"  flagged cases: {', '.join(f'[{i:02d}]' for i in flagged_indices)}")

    benchmark_latency(model, tokenizer)
    print()


if __name__ == "__main__":
    main()
