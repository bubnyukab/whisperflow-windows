"""Unit tests for the WhisperEngine wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.transcription.whisper_engine import WhisperEngine


class TestWhisperEngine:
    """Tests for WhisperEngine batch transcription logic."""

    def _make_engine(self, model: str = "tiny.en") -> WhisperEngine:
        engine = WhisperEngine.__new__(WhisperEngine)
        engine._model_name = model
        engine._model = MagicMock()
        return engine

    def test_returns_str(self) -> None:
        engine = self._make_engine()
        seg = MagicMock()
        seg.text = " hello world "
        engine._model.transcribe.return_value = ([seg], MagicMock())
        result = engine.transcribe("fake.wav")
        assert isinstance(result, str)

    def test_transcript_text_stripped(self) -> None:
        engine = self._make_engine()
        seg = MagicMock()
        seg.text = "  hello world  "
        engine._model.transcribe.return_value = ([seg], MagicMock())
        assert engine.transcribe("fake.wav") == "hello world"

    def test_multiple_segments_joined(self) -> None:
        engine = self._make_engine()
        segs = [MagicMock(text="Hello"), MagicMock(text="world")]
        engine._model.transcribe.return_value = (segs, MagicMock())
        assert engine.transcribe("fake.wav") == "Hello world"

    def test_empty_segments_returns_empty_string(self) -> None:
        engine = self._make_engine()
        engine._model.transcribe.return_value = ([], MagicMock())
        assert engine.transcribe("fake.wav") == ""

    def test_wav_path_passed_to_model(self) -> None:
        engine = self._make_engine()
        engine._model.transcribe.return_value = ([], MagicMock())
        engine.transcribe("/tmp/audio.wav")
        engine._model.transcribe.assert_called_once_with("/tmp/audio.wav", beam_size=5)

    def test_resolve_device_cpu_when_torch_absent(self) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            assert WhisperEngine._resolve_device("auto") == "cpu"

    def test_resolve_device_passthrough(self) -> None:
        assert WhisperEngine._resolve_device("cpu") == "cpu"
        assert WhisperEngine._resolve_device("cuda") == "cuda"

    def test_resolve_device_auto_with_cuda(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert WhisperEngine._resolve_device("auto") == "cuda"

    def test_resolve_device_auto_without_cuda(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert WhisperEngine._resolve_device("auto") == "cpu"
