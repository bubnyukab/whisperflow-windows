"""GPU and 4-bit quantization verification for QLoRA fine-tuning setup."""

import torch


def print_section(title: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


def check_cuda() -> None:
    print_section("CUDA / Torch")
    print(f"torch version   : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name        : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version    : {torch.version.cuda}")
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / 1024**3
        print(f"Total VRAM      : {total_vram:.2f} GB")
    else:
        print("No CUDA GPU detected — cannot proceed with 4-bit load.")


def get_free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(0)
    return free / 1024**3


def check_vram() -> None:
    print_section("VRAM (before model load)")
    if not torch.cuda.is_available():
        print("CUDA not available — skipping.")
        return
    free, total = torch.cuda.mem_get_info(0)
    print(f"Total VRAM      : {total / 1024**3:.2f} GB")
    print(f"Free  VRAM      : {free  / 1024**3:.2f} GB")
    print(f"Used  VRAM      : {(total - free) / 1024**3:.2f} GB")


def load_model_4bit() -> None:
    print_section("4-bit model load: Qwen/Qwen2.5-1.5B")
    if not torch.cuda.is_available():
        print("Skipping — CUDA not available.")
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        print(f"Import error: {e}")
        return

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_id = "Qwen/Qwen2.5-1.5B"
    print(f"Loading tokenizer from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    free_before = get_free_vram_gb()
    print(f"Loading model in 4-bit NF4 (free VRAM before: {free_before:.2f} GB) ...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("\nModel loaded successfully")

    print_section("VRAM (after model load)")
    free_after = get_free_vram_gb()
    _, total = torch.cuda.mem_get_info(0)
    used = total / 1024**3 - free_after
    vram_consumed = free_before - free_after
    print(f"Total VRAM      : {total / 1024**3:.2f} GB")
    print(f"Free  VRAM      : {free_after:.2f} GB")
    print(f"Used  VRAM      : {used:.2f} GB")
    print(f"Model used      : {vram_consumed:.2f} GB  (delta from before load)")

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params    : {num_params:.1f} M")


if __name__ == "__main__":
    check_cuda()
    check_vram()
    load_model_4bit()
    print("\nDone.")
