"""RealtimeSTT-backed continuous recorder for WhisperFlow."""

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
    """Continuous VAD-based recorder using RealtimeSTT + faster-whisper.

    The recorder loops in a daemon thread, calling AudioToTextRecorder.text(callback)
    which blocks until an utterance is complete, then invokes on_transcript.
    """

    def __init__(self, settings: Settings, on_transcript: Callable[[str], None]) -> None:
        self._settings = settings
        self._on_transcript = on_transcript
        self._recorder: Optional[object] = None
        self._running = False
        self._ready = False

    @property
    def is_ready(self) -> bool:
        """True after successful initialisation, False before start() or after stop()."""
        return self._ready

    def start(self) -> None:
        """Initialise AudioToTextRecorder and start the background loop thread."""
        if AudioToTextRecorder is None:  # pragma: no cover
            log.error("RealtimeSTT is not installed; cannot start recording")
            return
        try:
            self._recorder = AudioToTextRecorder(
                model=self._settings.whisper_model,
                language=self._settings.language,
                post_speech_silence_duration=self._settings.vad_silence_ms / 1000,
                silero_sensitivity=0.4,
                spinner=False,
                level=logging.WARNING,
            )
            self._ready = True
            self._running = True
            t = threading.Thread(target=self._loop, daemon=True)
            t.start()
            log.debug("RealtimeRecorder started (model=%s)", self._settings.whisper_model)
        except Exception:
            log.exception("RealtimeRecorder.start failed")

    def _loop(self) -> None:
        while self._running:
            try:
                self._recorder.text(self._on_transcript)  # type: ignore[union-attr]
            except Exception:
                if self._running:
                    log.exception("RealtimeRecorder loop error")
                break

    def stop(self) -> None:
        """Stop the recorder loop and release resources."""
        self._running = False
        self._ready = False
        recorder = self._recorder
        self._recorder = None
        if recorder is not None:
            try:
                recorder.shutdown()  # type: ignore[union-attr]
            except Exception:
                log.exception("RealtimeRecorder.stop shutdown failed")
