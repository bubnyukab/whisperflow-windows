"""Unit and integration tests for main.py."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to import from main with sys.argv controlled
# ---------------------------------------------------------------------------

def _parse(argv: list[str]):
    """Call _parse_args() with a controlled sys.argv."""
    from main import _parse_args
    with patch("sys.argv", ["whisperflow"] + argv):
        return _parse_args()


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_dev_false_by_default(self) -> None:
        assert _parse([]).dev is False

    def test_dev_true_when_flag_passed(self) -> None:
        assert _parse(["--dev"]).dev is True

    def test_model_none_by_default(self) -> None:
        assert _parse([]).model is None

    def test_model_override(self) -> None:
        assert _parse(["--model", "base.en"]).model == "base.en"

    def test_backend_none_by_default(self) -> None:
        assert _parse([]).backend is None

    def test_backend_fast_override(self) -> None:
        assert _parse(["--backend", "fast"]).backend == "fast"


# ---------------------------------------------------------------------------
# _apply_cli_overrides
# ---------------------------------------------------------------------------

class TestApplyCliOverrides:
    def _make_args(self, model=None, backend=None, dev=False):
        args = MagicMock()
        args.model = model
        args.backend = backend
        args.dev = dev
        return args

    def test_no_overrides_settings_unchanged(self) -> None:
        from main import _apply_cli_overrides
        from src.config.settings import Settings
        s = Settings()
        result = _apply_cli_overrides(self._make_args(), s)
        assert result.whisper_model == s.whisper_model
        assert result.formatter_backend == s.formatter_backend

    def test_model_override_applied(self) -> None:
        from main import _apply_cli_overrides
        from src.config.settings import Settings
        s = Settings()
        result = _apply_cli_overrides(self._make_args(model="medium.en"), s)
        assert result.whisper_model == "medium.en"

    def test_backend_override_applied(self) -> None:
        from main import _apply_cli_overrides
        from src.config.settings import Settings
        s = Settings()
        result = _apply_cli_overrides(self._make_args(backend="fast"), s)
        assert result.formatter_backend == "fast"

    def test_both_overrides_applied_together(self) -> None:
        from main import _apply_cli_overrides
        from src.config.settings import Settings
        s = Settings()
        result = _apply_cli_overrides(self._make_args(model="large-v3", backend="local"), s)
        assert result.whisper_model == "large-v3"
        assert result.formatter_backend == "local"

    def test_original_settings_not_mutated(self) -> None:
        from main import _apply_cli_overrides
        from src.config.settings import Settings
        s = Settings(whisper_model="tiny.en")
        _apply_cli_overrides(self._make_args(model="base.en"), s)
        assert s.whisper_model == "tiny.en"


# ---------------------------------------------------------------------------
# _append_history
# ---------------------------------------------------------------------------

class TestAppendHistory:
    def test_creates_file_when_not_exists(self, tmp_path: Path) -> None:
        from main import _append_history
        p = tmp_path / "history.json"
        _append_history({"text": "hello"}, p, 10)
        assert p.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        from main import _append_history
        p = tmp_path / "sub" / "dir" / "history.json"
        _append_history({"text": "hello"}, p, 10)
        assert p.exists()

    def test_entry_is_written(self, tmp_path: Path) -> None:
        from main import _append_history
        p = tmp_path / "history.json"
        _append_history({"raw_text": "hello", "final_text": "Hello."}, p, 10)
        data = json.loads(p.read_text())
        assert data[0]["raw_text"] == "hello"
        assert data[0]["final_text"] == "Hello."

    def test_multiple_entries_accumulate(self, tmp_path: Path) -> None:
        from main import _append_history
        p = tmp_path / "history.json"
        _append_history({"n": 1}, p, 10)
        _append_history({"n": 2}, p, 10)
        _append_history({"n": 3}, p, 10)
        data = json.loads(p.read_text())
        assert len(data) == 3
        assert data[-1]["n"] == 3

    def test_truncates_to_max_entries(self, tmp_path: Path) -> None:
        from main import _append_history
        p = tmp_path / "history.json"
        for i in range(5):
            _append_history({"n": i}, p, 3)
        data = json.loads(p.read_text())
        assert len(data) == 3
        assert data[0]["n"] == 2  # oldest kept
        assert data[-1]["n"] == 4  # newest

    def test_handles_corrupt_json_by_starting_fresh(self, tmp_path: Path) -> None:
        from main import _append_history
        p = tmp_path / "history.json"
        p.write_text("{{not valid json")
        _append_history({"text": "new"}, p, 10)
        data = json.loads(p.read_text())
        assert len(data) == 1
        assert data[0]["text"] == "new"

    def test_handles_non_list_json_by_starting_fresh(self, tmp_path: Path) -> None:
        from main import _append_history
        p = tmp_path / "history.json"
        p.write_text('{"not": "a list"}')
        _append_history({"text": "new"}, p, 10)
        data = json.loads(p.read_text())
        assert len(data) == 1


# ---------------------------------------------------------------------------
# _run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:
    """Tests for the synchronous pipeline function (format → inject → history)."""

    def _make_tray(self) -> MagicMock:
        tray = MagicMock()
        tray.set_state = MagicMock()
        tray.show_notification = MagicMock()
        return tray

    def test_sets_processing_state_before_format(self, tmp_path: Path) -> None:
        from main import _run_pipeline
        from src.config.settings import Settings
        tray = self._make_tray()
        injector = MagicMock()
        injector.inject.return_value = True
        calls: list[str] = []
        tray.set_state.side_effect = lambda s: calls.append(s)
        with patch("main.format_text", side_effect=lambda t, s: (calls.append("format"), t)[1]):
            _run_pipeline("hello", Settings(), tray, injector, tmp_path / "h.json")
        assert calls.index("processing") < calls.index("format")

    def test_calls_injector_with_formatted_text(self, tmp_path: Path) -> None:
        from main import _run_pipeline
        from src.config.settings import Settings
        tray = self._make_tray()
        injector = MagicMock()
        injector.inject.return_value = True
        with patch("main.format_text", return_value="Formatted text."):
            _run_pipeline("raw", Settings(), tray, injector, tmp_path / "h.json")
        injector.inject.assert_called_once_with("Formatted text.")

    def test_shows_fallback_toast_when_inject_fails(self, tmp_path: Path) -> None:
        from main import _run_pipeline
        from src.config.settings import Settings
        tray = self._make_tray()
        injector = MagicMock()
        injector.inject.return_value = False
        with patch("main.format_text", return_value="Text."):
            _run_pipeline("raw", Settings(), tray, injector, tmp_path / "h.json")
        injector.show_fallback_toast.assert_called_once_with("Text.")

    def test_no_fallback_toast_when_inject_succeeds(self, tmp_path: Path) -> None:
        from main import _run_pipeline
        from src.config.settings import Settings
        tray = self._make_tray()
        injector = MagicMock()
        injector.inject.return_value = True
        with patch("main.format_text", return_value="Text."):
            _run_pipeline("raw", Settings(), tray, injector, tmp_path / "h.json")
        injector.show_fallback_toast.assert_not_called()

    def test_appends_entry_to_history(self, tmp_path: Path) -> None:
        from main import _run_pipeline
        from src.config.settings import Settings
        tray = self._make_tray()
        injector = MagicMock()
        injector.inject.return_value = True
        history_path = tmp_path / "history.json"
        with patch("main.format_text", return_value="Final."):
            _run_pipeline("raw text", Settings(), tray, injector, history_path)
        data = json.loads(history_path.read_text())
        assert data[0]["raw_text"] == "raw text"
        assert data[0]["final_text"] == "Final."
        assert "timestamp" in data[0]

    def test_sets_done_state_after_pipeline(self, tmp_path: Path) -> None:
        from main import _run_pipeline
        from src.config.settings import Settings
        tray = self._make_tray()
        injector = MagicMock()
        injector.inject.return_value = True
        with patch("main.format_text", return_value="Text."):
            _run_pipeline("raw", Settings(), tray, injector, tmp_path / "h.json")
        last_state = tray.set_state.call_args_list[-1]
        assert last_state == call("done")

    def test_shows_tray_error_on_unexpected_exception(self, tmp_path: Path) -> None:
        from main import _run_pipeline
        from src.config.settings import Settings
        tray = self._make_tray()
        injector = MagicMock()
        with patch("main.format_text", side_effect=RuntimeError("boom")):
            _run_pipeline("raw", Settings(), tray, injector, tmp_path / "h.json")
        tray.show_notification.assert_called_once()

    def test_pipeline_never_raises(self, tmp_path: Path) -> None:
        from main import _run_pipeline
        from src.config.settings import Settings
        tray = self._make_tray()
        injector = MagicMock()
        injector.inject.side_effect = RuntimeError("inject crash")
        with patch("main.format_text", return_value="text"):
            _run_pipeline("raw", Settings(), tray, injector, tmp_path / "h.json")
        # If we reach here without exception, test passes


# ---------------------------------------------------------------------------
# Integration: full pipeline with sample.wav (transcription mocked)
# ---------------------------------------------------------------------------

SAMPLE_WAV = Path(__file__).parent / "fixtures" / "sample.wav"


class TestPipelineIntegration:
    """Dry-run integration: sample.wav → (mocked) transcription → format → (mocked) inject."""

    def test_sample_wav_fixture_exists(self) -> None:
        assert SAMPLE_WAV.exists(), "Run: python -c 'import wave...' to create fixture"

    def test_full_pipeline_writes_history(self, tmp_path: Path) -> None:
        """End-to-end pipeline with mocked transcription and injection."""
        from main import _run_pipeline
        from src.config.settings import Settings

        tray = MagicMock()
        tray.set_state = MagicMock()
        tray.show_notification = MagicMock()
        injector = MagicMock()
        injector.inject.return_value = True

        history_path = tmp_path / "history.json"
        raw = "um this is a test transcript you know"

        # Use real format_text (no mock) — tests actual formatting logic
        _run_pipeline(raw, Settings(formatter_backend="fast"), tray, injector, history_path)

        # History was written
        assert history_path.exists()
        data = json.loads(history_path.read_text())
        assert len(data) == 1
        assert data[0]["raw_text"] == raw
        # Formatter cleaned filler words
        assert "um" not in data[0]["final_text"]
        assert "you know" not in data[0]["final_text"]

    def test_full_pipeline_injects_formatted_text(self, tmp_path: Path) -> None:
        """Injector receives cleaned text, not the raw filler-laden transcript."""
        from main import _run_pipeline
        from src.config.settings import Settings

        tray = MagicMock()
        tray.set_state = MagicMock()
        injector = MagicMock()
        injector.inject.return_value = True

        raw = "um so basically this is a test"
        _run_pipeline(raw, Settings(formatter_backend="fast"), tray, injector,
                      tmp_path / "h.json")

        injected = injector.inject.call_args[0][0]
        assert "um" not in injected
        assert "basically" not in injected
        # Sentence starts capitalised
        assert injected[0].isupper()

    def test_pipeline_state_transitions_in_order(self, tmp_path: Path) -> None:
        """State machine goes processing → done (or processing → error)."""
        from main import _run_pipeline
        from src.config.settings import Settings

        states: list[str] = []
        tray = MagicMock()
        tray.set_state.side_effect = lambda s: states.append(s)
        injector = MagicMock()
        injector.inject.return_value = True

        _run_pipeline("test input", Settings(formatter_backend="fast"),
                      tray, injector, tmp_path / "h.json")

        assert states[0] == "processing"
        assert states[-1] == "done"
