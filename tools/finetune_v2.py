"""Warm-start QLoRA fine-tune: continue training from the v1 adapter.

Loads the existing adapter from models/whisperflow-cleaner/final, trains
for two more epochs at a lower learning rate on the 50K v2 dataset, and
saves to models/whisperflow-cleaner-v2/final.

Run from the project root:
    .venv/Scripts/python.exe tools/finetune_v2.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "tools" / "training_data_v2.jsonl"
V1_ADAPTER_DIR = ROOT / "models" / "whisperflow-cleaner" / "final"
OUTPUT_DIR = ROOT / "models" / "whisperflow-cleaner-v2"
ADAPTER_DIR = OUTPUT_DIR / "final"
MODEL_ID = "Qwen/Qwen2.5-1.5B"
SEED = 42
MAX_LENGTH = 384  # v2 sentences are longer (real wikipedia text)

random.seed(SEED)
torch.manual_seed(SEED)


def banner(text: str) -> None:
    print("=" * 64)
    print(f"  {text}")
    print("=" * 64)


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def tokenize_pair(example: dict, tokenizer, max_len: int = MAX_LENGTH) -> dict:
    prompt = f"Input: {example['input']}\nOutput: "
    completion = example["output"]

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
            labels.append(b["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info(0)
    return free / 1024**3


def main() -> None:
    banner("WhisperFlow v2 QLoRA fine-tune (warm start from v1 adapter)")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; QLoRA needs a GPU.")
    if not V1_ADAPTER_DIR.exists():
        raise FileNotFoundError(f"v1 adapter not found at {V1_ADAPTER_DIR}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"v2 training data not found at {DATA_PATH}")

    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"Free VRAM (start): {free_vram_gb():.2f} GB")
    print(f"v1 adapter       : {V1_ADAPTER_DIR}")
    print(f"v2 dataset       : {DATA_PATH}")

    # ── Data ──────────────────────────────────────────────────────────────────
    pairs = load_jsonl(DATA_PATH)
    print(f"\nLoaded {len(pairs):,} pairs")

    # ── Tokenizer (from v1 adapter dir, which has the saved tokenizer) ────────
    tokenizer = AutoTokenizer.from_pretrained(str(V1_ADAPTER_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Train/val split + tokenize ───────────────────────────────────────────
    ds = Dataset.from_list(pairs).train_test_split(test_size=0.05, seed=SEED)
    train_raw, val_raw = ds["train"], ds["test"]
    print(f"Train: {len(train_raw):,}   Val: {len(val_raw):,}")

    train_ds = train_raw.map(
        lambda x: tokenize_pair(x, tokenizer),
        remove_columns=train_raw.column_names,
        desc="tokenize train",
    )
    val_ds = val_raw.map(
        lambda x: tokenize_pair(x, tokenizer),
        remove_columns=val_raw.column_names,
        desc="tokenize val",
    )

    # ── Base model in 4-bit NF4, then attach existing adapter (trainable) ────
    print("\nLoading base model in 4-bit NF4 ...")
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
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    print(f"Attaching v1 adapter from {V1_ADAPTER_DIR} (trainable=True) ...")
    model = PeftModel.from_pretrained(base, str(V1_ADAPTER_DIR), is_trainable=True)
    model.print_trainable_parameters()
    print(f"Free VRAM (model loaded): {free_vram_gb():.2f} GB")

    # ── Training arguments ────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        warmup_ratio=0.02,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    banner("Starting training")
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    banner("Training complete")
    print(f"Final train loss : {train_result.training_loss:.4f}")
    print(f"Final eval loss  : {eval_metrics['eval_loss']:.4f}")

    # ── Save adapter ──────────────────────────────────────────────────────────
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"\nLoRA adapter saved -> {ADAPTER_DIR}")

    # ── 5 validation examples ─────────────────────────────────────────────────
    banner("5 validation examples (model output vs gold)")
    model.config.use_cache = True
    model.eval()
    samples = random.sample(list(val_raw), min(5, len(val_raw)))
    for i, sample in enumerate(samples, 1):
        prompt = f"Input: {sample['input']}\nOutput: "
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        plen = ids.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][plen:], skip_special_tokens=True)
        gen = gen.split("\n")[0].strip()
        print(f"\n[{i}]")
        print(f"  INPUT     : {sample['input']}")
        print(f"  GOLD      : {sample['output']}")
        print(f"  PREDICTED : {gen}")
    print()


if __name__ == "__main__":
    main()
