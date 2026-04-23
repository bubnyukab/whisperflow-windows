"""Unit tests for the RealtimeRecorder wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config.settings import Settings
from src.audio.realtime_recorder import RealtimeRecorder


class TestRealtimeRecorder:
    """Tests for push-to-talk recording lifecycle."""

    def setup_method(self) -> None:
        self.settings = Settings()
        self.received: list[str] = []
        self.recorder = RealtimeRecorder(self.settings, on_text=self.received.append)

    def test_end_session_without_start_is_noop(self) -> None:
        """end_session when no session is active should not raise."""
        self.recorder.end_session()
        assert self.received == []

    def test_shutdown_safe_when_idle(self) -> None:
        """shutdown() should not raise when no recorder is active."""
        self.recorder.shutdown()

    def test_on_text_callback_receives_transcript(self) -> None:
        """Transcript text is forwarded to on_text callback after end_session."""
        mock_instance = MagicMock()
        mock_instance.text.return_value = "hello world"

        with patch("src.audio.realtime_recorder.AudioToTextRecorder", return_value=mock_instance):
            self.recorder.start_session()

        self.recorder.end_session()
        assert self.received == ["hello world"]

    def test_empty_transcript_not_forwarded(self) -> None:
        """Blank transcripts should not trigger the on_text callback."""
        mock_instance = MagicMock()
        mock_instance.text.return_value = "   "

        with patch("src.audio.realtime_recorder.AudioToTextRecorder", return_value=mock_instance):
            self.recorder.start_session()

        self.recorder.end_session()
        assert self.received == []
