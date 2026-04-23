"""RealtimeSTT-backed push-to-talk recorder for WhisperFlow."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from src.config.settings import Settings

log = logging.getLogger(__name__)

try:
    from RealtimeSTT import AudioToTextRecorder
except ImportError:  # pragma: no cover
    AudioToTextRecorder = None  # type: ignore[assignment,misc]


class RealtimeRecorder:
    """Push-to-talk audio recorder using RealtimeSTT with VAD and faster-whisper."""

    def __init__(self, settings: Settings, on_text: Callable[[str], None]) -> None:
        """Initialize the recorder.

        Args:
            settings: Application settings (whisper model, etc.)
            on_text: Callback invoked with the transcribed text once recording ends.
        """
        self._settings = settings
        self._on_text = on_text
        self._recorder: Optional[object] = None
        self._lock = threading.Lock()

    def start_session(self) -> None:
        """Start a recording session (called on hotkey press)."""
        with self._lock:
            if self._recorder is not None:
                return
            if AudioToTextRecorder is None:
                log.error("RealtimeSTT is not installed; cannot start recording")
                return
            try:
                self._recorder = AudioToTextRecorder(
                    model=self._settings.whisper_model,
                    spinner=False,
                    use_microphone=True,
                    level=logging.WARNING,
                )
                self._recorder.start()  # type: ignore[union-attr]
                log.debug("Recording session started")
            except Exception:
                log.exception("Failed to start recording session")
                self._recorder = None

    def end_session(self) -> None:
        """End the recording session (called on hotkey release) and emit transcript."""
        with self._lock:
            if self._recorder is None:
                return
            try:
                text: str = self._recorder.text() or ""  # type: ignore[union-attr]
                self._recorder.stop()  # type: ignore[union-attr]
                log.debug("Session ended, transcript: %r", text)
            except Exception:
                log.exception("Failed to end recording session")
                text = ""
            finally:
                self._recorder = None

        if text.strip():
            self._on_text(text.strip())

    def shutdown(self) -> None:
        """Release all resources."""
        with self._lock:
            if self._recorder is not None:
                try:
                    self._recorder.stop()  # type: ignore[union-attr]
                except Exception:
                    pass
                self._recorder = None
