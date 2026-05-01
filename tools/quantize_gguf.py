"""Quantize an fp16 GGUF cleaner to Q4_K_M using libllama via llama-cpp-python.

Run from the project root:
    .venv/Scripts/python.exe tools/quantize_gguf.py
    .venv/Scripts/python.exe tools/quantize_gguf.py \
        --src models/whisperflow-cleaner-v2/model-fp16.gguf \
        --dst models/whisperflow-cleaner-v2/model.gguf
"""

from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

# torch must load first so its bundled CUDA DLLs are in the process before
# llama_cpp probes for cudart64_12.dll / cublas64_12.dll. Even though the
# quantize step itself is CPU-only, libllama is loaded with CUDA bindings
# and will fail to import without these DLLs resolved.
import torch  # noqa: F401
from llama_cpp import llama_cpp as lc

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "models" / "whisperflow-cleaner" / "model-fp16.gguf"
DEFAULT_DST = ROOT / "models" / "whisperflow-cleaner" / "model.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize fp16 GGUF to Q4_K_M.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help="Input fp16 GGUF path")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST,
                        help="Output Q4_K_M GGUF path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise FileNotFoundError(f"Missing source GGUF: {args.src}")

    print(f"Source : {args.src}  ({args.src.stat().st_size / 1024**3:.2f} GB)")
    print(f"Target : {args.dst}")
    print(f"Type   : Q4_K_M (ftype={lc.LLAMA_FTYPE_MOSTLY_Q4_K_M})")

    args.dst.parent.mkdir(parents=True, exist_ok=True)

    params = lc.llama_model_quantize_default_params()
    params.ftype = lc.LLAMA_FTYPE_MOSTLY_Q4_K_M

    t0 = time.perf_counter()
    ret = lc.llama_model_quantize(
        str(args.src).encode("utf-8"),
        str(args.dst).encode("utf-8"),
        ctypes.byref(params),
    )
    dt = time.perf_counter() - t0

    if ret != 0:
        raise RuntimeError(f"llama_model_quantize returned {ret}")

    print(f"\nDone in {dt:.1f}s. On-disk size: {args.dst.stat().st_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
