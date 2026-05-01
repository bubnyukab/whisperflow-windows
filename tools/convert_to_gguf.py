"""Convert a merged HF checkpoint directory to fp16 GGUF.

Wraps the convert_hf_to_gguf.py script that ships with llama-cpp-python in
`.venv/Lib/site-packages/bin/`. Output is the fp16 GGUF the quantize step
consumes.

Run from the project root:
    .venv/Scripts/python.exe tools/convert_to_gguf.py
    .venv/Scripts/python.exe tools/convert_to_gguf.py \
        --src models/whisperflow-cleaner-v2/merged-fp16 \
        --dst models/whisperflow-cleaner-v2/model-fp16.gguf
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_BIN = ROOT / ".venv" / "Lib" / "site-packages" / "bin"
CONVERTER = VENV_BIN / "convert_hf_to_gguf.py"

DEFAULT_SRC = ROOT / "models" / "whisperflow-cleaner" / "merged-fp16"
DEFAULT_DST = ROOT / "models" / "whisperflow-cleaner" / "model-fp16.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert merged HF checkpoint to fp16 GGUF.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help="Path to merged HF checkpoint directory")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST,
                        help="Path to output fp16 GGUF file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise FileNotFoundError(f"Missing merged checkpoint: {args.src}")
    if not CONVERTER.exists():
        raise FileNotFoundError(f"Missing converter script: {CONVERTER}")

    args.dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(CONVERTER),
        str(args.src),
        "--outfile", str(args.dst),
        "--outtype", "f16",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"convert_hf_to_gguf.py exited with {result.returncode}")

    size_gb = args.dst.stat().st_size / 1024**3
    print(f"\nDone. fp16 GGUF written: {args.dst}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
