"""Local GGUF LLM formatter — runs the fine-tuned Qwen2.5 cleaner via llama-cpp-python.

The model is a Q4_K_M-quantised Qwen2.5-1.5B with our LoRA adapter merged in,
trained on the Input:/Output: completion pattern. On an RTX 3060 it averages
~105 ms per cleanup with full GPU offload — well under the 400 ms budget.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = "Input: {text}\nOutput: "
_STOP = ["\n"]
_MAX_TOKENS = 200

_INSTALL_HINT = (
    "llama-cpp-python is not installed. Run:\n"
    "  pip install llama-cpp-python==0.3.4 "
    "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
)


def _load_llama_class():
    """Import torch then llama_cpp, raising RuntimeError with install hint if missing.

    torch must be imported first on Windows so its bundled CUDA DLLs are in the
    loader search path before libllama.dll tries to resolve cudart64_12.dll etc.
    """
    try:
        import torch  # noqa: F401  # must precede llama_cpp on Windows
        from llama_cpp import Llama
        return Llama
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc


class LocalLLMFormatter:
    """GGUF-backed transcript cleaner. Loads once on construction; subsequent
    .format() calls are ~100 ms on an RTX 3060."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_gpu_layers: int = -1,
        n_ctx: int = 512,
        n_batch: int = 512,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Local model not found at {path}. "
                "Run tools/finetune.py to train the model first."
            )
        Llama = _load_llama_class()
        log.info("Loading GGUF cleaner from %s", path)
        self._model_path = path
        self._llm = Llama(
            model_path=str(path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
            n_batch=n_batch,
            seed=0,
            verbose=False,
        )
        log.info("LocalLLMFormatter loaded successfully from %s", path)

    def format(self, raw_transcript: str, context_hint: str = "") -> str:
        """Clean a transcript via the local GGUF model. Falls back to raw on any error."""
        if not raw_transcript:
            return ""
        prompt = _PROMPT_TEMPLATE.format(text=raw_transcript)
        try:
            out = self._llm.create_completion(
                prompt,
                max_tokens=_MAX_TOKENS,
                temperature=0.0,
                stop=_STOP,
                echo=False,
            )
            text = out["choices"][0]["text"].strip()
            return text or raw_transcript
        except Exception:
            log.exception("LocalLLMFormatter.format failed")
            return raw_transcript

    def is_available(self) -> bool:
        """LocalLLMFormatter loads the model in __init__ — if we exist, we're ready."""
        return True
