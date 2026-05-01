"""Merge a QLoRA adapter into the fp16 base model and save to disk.

Output: <output_dir>/  (~3 GB on disk)
The merged model can then be loaded as a vanilla HF checkpoint with no
PEFT/bitsandbytes runtime dependency.

Run from the project root:
    .venv/Scripts/python.exe tools/merge_adapter.py
    .venv/Scripts/python.exe tools/merge_adapter.py \
        --adapter-dir models/whisperflow-cleaner-v2/final \
        --output-dir  models/whisperflow-cleaner-v2/merged-fp16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER = ROOT / "models" / "whisperflow-cleaner" / "final"
DEFAULT_OUTPUT = ROOT / "models" / "whisperflow-cleaner" / "merged-fp16"
MODEL_ID = "Qwen/Qwen2.5-1.5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into fp16 base.")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER,
                        help="Directory containing adapter_model.safetensors")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to write the merged fp16 HF checkpoint")
    parser.add_argument("--model-id", type=str, default=MODEL_ID,
                        help="Base model HF id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info(0)[0] / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)}  free VRAM: {free_gb:.2f} GB")
    else:
        print("GPU: not available - merging on CPU")

    print(f"\nLoading base model {args.model_id} in fp16 ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Attaching adapter from {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(base, str(args.adapter_dir))

    print("Merging adapter weights into base ...")
    model = model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {args.output_dir} ...")
    model.save_pretrained(str(args.output_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(args.output_dir))

    total = sum(p.stat().st_size for p in args.output_dir.rglob("*") if p.is_file())
    print(f"\nDone. On-disk size: {total / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
