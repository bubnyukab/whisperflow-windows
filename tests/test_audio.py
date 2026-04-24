"""Unit tests for the RealtimeRecorder wrapper."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings
from src.audio.realtime_recorder import RealtimeRecorder


class TestRealtimeRecorder:
    """Tests for the VAD-based continuous recorder lifecycle."""

    def test_is_ready_false_before_start(self) -> None:
        recorder = RealtimeRecorder(Settings(), on_transcript=lambda t: None)
        assert recorder.is_ready is False

    def test_is_ready_true_after_start(self) -> None:
        mock_instance = MagicMock()
        recorder = RealtimeRecorder(Settings(), on_transcript=lambda t: None)
        mock_instance.text.side_effect = lambda cb: setattr(recorder, "_running", False)

        with patch("src.audio.realtime_recorder.AudioToTextRecorder", return_value=mock_instance):
            recorder.start()

        assert recorder.is_ready is True
        recorder.stop()

    def test_is_ready_false_after_stop(self) -> None:
        mock_instance = MagicMock()
        recorder = RealtimeRecorder(Settings(), on_transcript=lambda t: None)
        mock_instance.text.side_effect = lambda cb: setattr(recorder, "_running", False)

        with patch("src.audio.realtime_recorder.AudioToTextRecorder", return_value=mock_instance):
            recorder.start()

        recorder.stop()
        assert recorder.is_ready is False

    def test_start_passes_correct_params_to_recorder(self) -> None:
        settings = Settings(whisper_model="base.en", language="en", vad_silence_ms=600)
        mock_instance = MagicMock()
        recorder = RealtimeRecorder(settings, on_transcript=lambda t: None)
        mock_instance.text.side_effect = lambda cb: setattr(recorder, "_running", False)

        with patch(
            "src.audio.realtime_recorder.AudioToTextRecorder", return_value=mock_instance
        ) as mock_cls:
            recorder.start()

        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "base.en"
        assert kwargs["language"] == "en"
        assert kwargs["post_speech_silence_duration"] == pytest.approx(0.6)
        assert kwargs["silero_sensitivity"] == 0.4
        assert kwargs["spinner"] is False
        recorder.stop()

    def test_on_transcript_called_with_text(self) -> None:
        received: list[str] = []
        done = threading.Event()
        recorder: RealtimeRecorder  # forward ref for closure

        def fake_text(callback: object) -> None:
            callback("hello world")  # type: ignore[operator]
            done.set()
            recorder._running = False

        mock_instance = MagicMock()
        mock_instance.text.side_effect = fake_text

        with patch("src.audio.realtime_recorder.AudioToTextRecorder", return_value=mock_instance):
            recorder = RealtimeRecorder(Settings(), on_transcript=received.append)
            recorder.start()

        done.wait(timeout=2.0)
        assert received == ["hello world"]
        recorder.stop()

    def test_stop_calls_shutdown(self) -> None:
        mock_instance = MagicMock()
        recorder = RealtimeRecorder(Settings(), on_transcript=lambda t: None)
        mock_instance.text.side_effect = lambda cb: setattr(recorder, "_running", False)

        with patch("src.audio.realtime_recorder.AudioToTextRecorder", return_value=mock_instance):
            recorder.start()

        recorder.stop()
        mock_instance.shutdown.assert_called_once()

    def test_stop_before_start_is_safe(self) -> None:
        recorder = RealtimeRecorder(Settings(), on_transcript=lambda t: None)
        recorder.stop()
        assert recorder.is_ready is False

    def test_start_when_lib_missing_is_noop(self) -> None:
        recorder = RealtimeRecorder(Settings(), on_transcript=lambda t: None)
        with patch("src.audio.realtime_recorder.AudioToTextRecorder", None):
            recorder.start()
        assert recorder.is_ready is False

    def test_multiple_transcripts_forwarded(self) -> None:
        received: list[str] = []
        call_count = 0
        done = threading.Event()
        recorder: RealtimeRecorder

        def fake_text(callback: object) -> None:
            nonlocal call_count
            call_count += 1
            callback(f"utterance {call_count}")  # type: ignore[operator]
            if call_count >= 2:
                done.set()
                recorder._running = False

        mock_instance = MagicMock()
        mock_instance.text.side_effect = fake_text

        with patch("src.audio.realtime_recorder.AudioToTextRecorder", return_value=mock_instance):
            recorder = RealtimeRecorder(Settings(), on_transcript=received.append)
            recorder.start()

        done.wait(timeout=2.0)
        assert received == ["utterance 1", "utterance 2"]
        recorder.stop()
