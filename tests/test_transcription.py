"""Unit tests for the WhisperEngine wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.transcription.whisper_engine import TranscriptionResult, WhisperEngine


class TestWhisperEngine:
    """Tests for WhisperEngine transcription logic."""

    def test_transcription_result_type(self) -> None:
        """transcribe() should return a TranscriptionResult with correct fields."""
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = " hello world "
        mock_model.transcribe.return_value = ([mock_segment], MagicMock())

        with patch("src.transcription.whisper_engine.WhisperModel", return_value=mock_model):
            engine = WhisperEngine.__new__(WhisperEngine)
            engine._model_name = "tiny.en"
            engine._model = mock_model
            result = engine.transcribe(Path("fake.wav"))

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        assert result.model == "tiny.en"
        assert result.duration_ms >= 0

    def test_resolve_device_cpu_when_torch_absent(self) -> None:
        """auto device resolves to cpu when torch is not installed."""
        with patch.dict("sys.modules", {"torch": None}):
            assert WhisperEngine._resolve_device("auto") == "cpu"

    def test_resolve_device_explicit(self) -> None:
        """Explicit device strings are passed through unchanged."""
        assert WhisperEngine._resolve_device("cpu") == "cpu"
        assert WhisperEngine._resolve_device("cuda") == "cuda"
