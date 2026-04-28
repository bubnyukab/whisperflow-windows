"""Local GGUF LLM formatter — runs the fine-tuned Qwen2.5 cleaner via llama-cpp-python.

The model is a Q4_K_M-quantised Qwen2.5-1.5B with our LoRA adapter merged in,
trained on the Input:/Output: completion pattern. On an RTX 3060 it averages
~105 ms per cleanup with full GPU offload — well under the 400 ms budget.
"""

from __future__ import annotations

import logging
from pathlib import Path

# torch must be imported before llama_cpp on Windows. The cu124 prebuilt
# llama-cpp-python wheel needs cudart64_12.dll, cublas64_12.dll, and
# nvrtc64_120_0.dll resolved when libllama.dll is loaded; torch's bundled
# CUDA runtime in torch/lib/ is what provides them. Without this import
# order, `from llama_cpp import Llama` fails with FileNotFoundError on
# llama.dll's transitive dependencies.
import torch  # noqa: F401
from llama_cpp import Llama

log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = "Input: {text}\nOutput: "
_STOP = ["\n"]
_MAX_TOKENS = 200


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
            raise FileNotFoundError(f"GGUF model not found: {path}")
        log.info("Loading GGUF cleaner from %s", path)
        self._model_path = path
        self._llm = Llama(
            model_path=str(path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            seed=0,
            verbose=False,
        )

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
