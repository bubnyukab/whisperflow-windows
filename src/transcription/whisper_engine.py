"""Direct faster-whisper engine wrapper — used by the benchmark tool and CLI mode."""

from __future__ import annotations


class WhisperEngine:
    """Thin wrapper around faster_whisper.WhisperModel for batch transcription."""

    def __init__(self, model: str = "tiny.en", device: str = "auto") -> None:
        from faster_whisper import WhisperModel

        resolved_device = self._resolve_device(device)
        self._model_name = model
        self._model = WhisperModel(model, device=resolved_device, compute_type="auto")

    def transcribe(self, wav_path: str) -> str:
        """Transcribe an audio file and return the transcript text."""
        segments, _ = self._model.transcribe(wav_path, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
