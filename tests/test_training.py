"""Tests for src/training/collector.py and src/training/review_window.py."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.training.collector import TrainingCollector


# ---------------------------------------------------------------------------
# TrainingCollector
# ---------------------------------------------------------------------------

class TestTrainingCollector:
    def test_save_pair_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "pairs.jsonl"
        c = TrainingCollector(p)
        c.save_pair("raw", "clean")
        assert p.exists()

    def test_save_pair_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "pairs.jsonl"
        TrainingCollector(p).save_pair("a", "b")
        assert p.exists()

    def test_save_pair_writes_jsonl_line(self, tmp_path: Path) -> None:
        p = tmp_path / "pairs.jsonl"
        TrainingCollector(p).save_pair("hello world", "Hello world.")
        line = p.read_text(encoding="utf-8").strip()
        obj = json.loads(line)
        assert obj["input"] == "hello world"
        assert obj["output"] == "Hello world."

    def test_save_pair_includes_timestamp(self, tmp_path: Path) -> None:
        p = tmp_path / "pairs.jsonl"
        TrainingCollector(p).save_pair("x", "y")
        obj = json.loads(p.read_text(encoding="utf-8").strip())
        assert "timestamp" in obj
        assert len(obj["timestamp"]) > 0

    def test_multiple_saves_append_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "pairs.jsonl"
        c = TrainingCollector(p)
        c.save_pair("a", "A.")
        c.save_pair("b", "B.")
        c.save_pair("c", "C.")
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 3

    def test_save_pair_never_raises_on_bad_path(self) -> None:
        c = TrainingCollector(Path("/this/path/does/not/exist/x/y/z.jsonl"))
        # Should not raise — errors are logged only
        with patch("src.training.collector.Path.parent") as mock_parent:
            mock_parent.mkdir.side_effect = PermissionError("denied")
            # Re-create with a real bad path scenario: just verify no exception escapes
        # Use the real path but deny mkdir by patching it differently
        with patch.object(Path, "mkdir", side_effect=PermissionError("no")):
            TrainingCollector(Path("/dev/null/bad.jsonl")).save_pair("x", "y")

    def test_load_pairs_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        p = tmp_path / "pairs.jsonl"
        assert TrainingCollector(p).load_pairs() == []

    def test_load_pairs_returns_all_saved(self, tmp_path: Path) -> None:
        p = tmp_path / "pairs.jsonl"
        c = TrainingCollector(p)
        c.save_pair("first", "First.")
        c.save_pair("second", "Second.")
        pairs = c.load_pairs()
        assert len(pairs) == 2
        assert pairs[0]["input"] == "first"
        assert pairs[1]["input"] == "second"

    def test_load_pairs_skips_corrupt_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "pairs.jsonl"
        p.write_text('{"input": "ok", "output": "Ok."}\n{bad json}\n', encoding="utf-8")
        pairs = TrainingCollector(p).load_pairs()
        assert len(pairs) == 1
        assert pairs[0]["input"] == "ok"

    def test_load_pairs_skips_blank_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "pairs.jsonl"
        p.write_text('\n{"input": "x", "output": "y"}\n\n', encoding="utf-8")
        pairs = TrainingCollector(p).load_pairs()
        assert len(pairs) == 1

    def test_save_then_load_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "pairs.jsonl"
        c = TrainingCollector(p)
        c.save_pair("um yeah so basically", "Yeah.")
        loaded = c.load_pairs()
        assert loaded[0]["input"] == "um yeah so basically"
        assert loaded[0]["output"] == "Yeah."


# ---------------------------------------------------------------------------
# ReviewWindow — unit tests (no GUI rendered)
# ---------------------------------------------------------------------------

class TestReviewWindowInit:
    def test_stores_raw_and_cleaned(self) -> None:
        from src.training.review_window import ReviewWindow
        collector = MagicMock()
        w = ReviewWindow("raw text", "Clean text.", collector)
        assert w._raw == "raw text"
        assert w._cleaned == "Clean text."

    def test_stores_collector(self) -> None:
        from src.training.review_window import ReviewWindow
        collector = MagicMock()
        w = ReviewWindow("r", "c", collector)
        assert w._collector is collector


class TestReviewWindowRun:
    """Test that run() calls build_and_run and handles exceptions gracefully."""

    def test_run_does_not_raise_when_tkinter_fails(self) -> None:
        from src.training.review_window import ReviewWindow
        collector = MagicMock()
        w = ReviewWindow("raw", "clean", collector)
        with patch.object(w, "_build_and_run", side_effect=RuntimeError("tk not available")):
            w.run()  # must not raise

    def test_accept_saves_pair_as_is(self, tmp_path: Path) -> None:
        """_accept() saves raw→cleaned without modification."""
        from src.training.review_window import ReviewWindow
        p = tmp_path / "pairs.jsonl"
        collector = TrainingCollector(p)
        w = ReviewWindow("raw", "cleaned", collector)
        # Simulate having a root with after_cancel and destroy
        mock_root = MagicMock()
        w._root = mock_root
        w._timer_id = "t1"
        w._accept()
        mock_root.destroy.assert_called_once()
        pairs = collector.load_pairs()
        assert len(pairs) == 1
        assert pairs[0]["input"] == "raw"
        assert pairs[0]["output"] == "cleaned"

    def test_skip_does_not_save(self, tmp_path: Path) -> None:
        from src.training.review_window import ReviewWindow
        p = tmp_path / "pairs.jsonl"
        collector = TrainingCollector(p)
        w = ReviewWindow("raw", "cleaned", collector)
        mock_root = MagicMock()
        w._root = mock_root
        w._timer_id = None
        w._skip()
        mock_root.destroy.assert_called_once()
        assert not p.exists()

    def test_save_correction_saves_edited_text(self, tmp_path: Path) -> None:
        from src.training.review_window import ReviewWindow
        p = tmp_path / "pairs.jsonl"
        collector = TrainingCollector(p)
        w = ReviewWindow("raw", "cleaned", collector)
        mock_root = MagicMock()
        mock_edit_box = MagicMock()
        mock_edit_box.get.return_value = "user edited text\n"
        w._root = mock_root
        w._edit_box = mock_edit_box
        w._timer_id = None
        w._save_correction()
        mock_root.destroy.assert_called_once()
        pairs = collector.load_pairs()
        assert pairs[0]["output"] == "user edited text"

    def test_save_correction_falls_back_to_cleaned_when_edit_empty(
        self, tmp_path: Path
    ) -> None:
        from src.training.review_window import ReviewWindow
        p = tmp_path / "pairs.jsonl"
        collector = TrainingCollector(p)
        w = ReviewWindow("raw", "original clean", collector)
        mock_root = MagicMock()
        mock_edit_box = MagicMock()
        mock_edit_box.get.return_value = "   "  # whitespace only
        w._root = mock_root
        w._edit_box = mock_edit_box
        w._timer_id = None
        w._save_correction()
        pairs = collector.load_pairs()
        assert pairs[0]["output"] == "original clean"


# ---------------------------------------------------------------------------
# Pipeline integration: training mode flag check in _run_pipeline
# ---------------------------------------------------------------------------

class TestPipelineTrainingMode:
    def _make_tray(self, training_mode: bool = False) -> MagicMock:
        tray = MagicMock()
        tray.set_state = MagicMock()
        tray.show_notification = MagicMock()
        tray.training_mode = training_mode
        return tray

    def test_review_window_spawned_when_training_mode_on(self, tmp_path: Path) -> None:
        from main import _run_pipeline
        from src.config.settings import Settings

        tray = self._make_tray(training_mode=True)
        injector = MagicMock()
        injector.inject.return_value = True
        collector = MagicMock()

        spawned: list[threading.Thread] = []
        real_thread = threading.Thread

        def capture_thread(*args, **kwargs):
            t = real_thread(*args, **kwargs)
            spawned.append(t)
            return t

        with patch("main.format_text", return_value="Formatted."), \
             patch("main.threading.Thread", side_effect=capture_thread):
            _run_pipeline("raw", Settings(), tray, injector,
                          tmp_path / "h.json", collector)

        # At least one daemon thread was spawned for the review window
        assert any(t.daemon for t in spawned)

    def test_review_window_not_spawned_when_training_mode_off(self, tmp_path: Path) -> None:
        from main import _run_pipeline
        from src.config.settings import Settings

        tray = self._make_tray(training_mode=False)
        injector = MagicMock()
        injector.inject.return_value = True
        collector = MagicMock()

        with patch("main.format_text", return_value="Formatted."):
            _run_pipeline("raw", Settings(), tray, injector,
                          tmp_path / "h.json", collector)

        collector.save_pair.assert_not_called()

    def test_review_window_not_spawned_when_no_collector(self, tmp_path: Path) -> None:
        from main import _run_pipeline
        from src.config.settings import Settings

        tray = self._make_tray(training_mode=True)
        injector = MagicMock()
        injector.inject.return_value = True

        with patch("main.format_text", return_value="Formatted."):
            _run_pipeline("raw", Settings(), tray, injector,
                          tmp_path / "h.json", collector=None)
        # Should not raise

    def test_build_on_transcript_passes_collector(self) -> None:
        from main import _build_on_transcript

        tray = MagicMock()
        tray._settings = MagicMock()
        injector = MagicMock()
        collector = MagicMock()

        spawned_args: list = []
        with patch("main.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            cb = _build_on_transcript(tray, injector, Path("/tmp/h.json"), collector)
            cb("hello")
            _, kwargs = mock_thread.call_args
            assert kwargs["args"][-1] is collector
