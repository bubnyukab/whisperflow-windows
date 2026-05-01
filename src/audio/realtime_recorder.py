"""RealtimeSTT-backed recorder for WhisperFlow — hold-to-talk mode."""

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
    """Hold-to-talk wrapper around AudioToTextRecorder.

    Correct lifecycle:
        recorder = RealtimeRecorder(settings, on_transcript)
        recorder.initialize()   # once at startup, in a background thread
        recorder.start()        # hotkey pressed  → begin capturing audio
        recorder.stop()         # hotkey released → transcribe, fire on_transcript
        recorder.shutdown()     # app exit only   → destroy subprocess
    """

    def __init__(
        self,
        settings: Settings,
        on_transcript: Callable[[str], None],
        on_init_failed: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._settings = settings
        self._on_transcript = on_transcript
        self._on_init_failed = on_init_failed
        self._recorder: Optional[object] = None
        self._ready = False
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create AudioToTextRecorder — call once, in a background thread at startup.

        This is the expensive step (~15-20 s on first run while the model downloads;
        ~1-2 s on subsequent runs once cached).  is_ready becomes True when done.
        """
        if AudioToTextRecorder is None:  # pragma: no cover
            log.error("RealtimeSTT is not installed; recording unavailable")
            return
        try:
            recorder = AudioToTextRecorder(
                model=self._settings.whisper_model,
                language=self._settings.language,
                post_speech_silence_duration=self._settings.vad_silence_ms / 1000,
                silero_sensitivity=0.4,
                compute_type="int8",  # fastest on CPU; eliminates the float32 fallback warning
                beam_size=1,          # greedy decoding — ~30% faster, imperceptible accuracy loss
                spinner=False,
                level=logging.WARNING,
            )
            with self._lock:
                self._recorder = recorder
                self._ready = True
            log.debug("RealtimeRecorder ready (model=%s)", self._settings.whisper_model)
        except Exception:
            log.exception("RealtimeRecorder.initialize failed — recording unavailable")
            if self._on_init_failed is not None:
                self._on_init_failed(
                    "Whisper failed to load — recording unavailable. Check logs for details."
                )

    def shutdown(self) -> None:
        """Destroy the recorder subprocess — call only on app exit."""
        with self._lock:
            recorder = self._recorder
            self._recorder = None
            self._ready = False
        if recorder is not None:
            try:
                recorder.shutdown()  # type: ignore[union-attr]
            except Exception:
                log.exception("RealtimeRecorder.shutdown failed")

    # ------------------------------------------------------------------
    # Per-recording controls (called on every hotkey press/release)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin audio capture (hotkey pressed)."""
        with self._lock:
            recorder = self._recorder
        if recorder is not None:
            try:
                recorder.start()  # type: ignore[union-attr]
                log.debug("RealtimeRecorder: capture started")
            except Exception:
                log.exception("RealtimeRecorder.start failed")

    def stop(self) -> None:
        """End audio capture and transcribe in background (hotkey released)."""
        with self._lock:
            recorder = self._recorder
        if recorder is None:
            return
        try:
            recorder.stop()  # type: ignore[union-attr]
            threading.Thread(target=self._transcribe, args=(recorder,), daemon=True).start()
        except Exception:
            log.exception("RealtimeRecorder.stop failed")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transcribe(self, recorder: object) -> None:
        """Block on recorder.text(), then fire on_transcript with the result."""
        try:
            text = recorder.text()  # type: ignore[union-attr]
            if text and text.strip():
                self._on_transcript(text.strip())
        except Exception:
            log.exception("RealtimeRecorder._transcribe failed")
