"""Direct faster-whisper engine wrapper — used by the benchmark tool and unit tests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscriptionResult:
    """Result from a single transcription pass."""

    text: str
    duration_ms: float
    model: str


class WhisperEngine:
    """Thin wrapper around faster_whisper.WhisperModel."""

    def __init__(self, model: str = "tiny.en", device: str = "auto") -> None:
        """Load and cache the Whisper model.

        Args:
            model: Model name (tiny.en, base.en, medium.en, large-v3).
            device: 'cuda', 'cpu', or 'auto' (detects CUDA availability).
        """
        from faster_whisper import WhisperModel

        resolved_device = self._resolve_device(device)
        self._model_name = model
        self._model = WhisperModel(model, device=resolved_device, compute_type="auto")

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file and return the result.

        Args:
            audio_path: Path to a WAV or MP3 file.
        """
        t0 = time.perf_counter()
        segments, _ = self._model.transcribe(str(audio_path), beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
        elapsed = (time.perf_counter() - t0) * 1000
        return TranscriptionResult(text=text, duration_ms=elapsed, model=self._model_name)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu' based on availability."""
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
