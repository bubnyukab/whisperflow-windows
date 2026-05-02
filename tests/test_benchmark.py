"""Unit tests for tools/benchmark.py."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure tools/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# _time_ms
# ---------------------------------------------------------------------------

class TestTimeMs:
    def test_returns_result_and_float(self) -> None:
        from tools.benchmark import _time_ms
        result, elapsed = _time_ms(lambda: 42)
        assert result == 42
        assert isinstance(elapsed, float)

    def test_elapsed_is_positive(self) -> None:
        from tools.benchmark import _time_ms
        _, elapsed = _time_ms(lambda: None)
        assert elapsed >= 0.0

    def test_elapsed_reflects_real_time(self) -> None:
        from tools.benchmark import _time_ms
        def slow():
            time.sleep(0.05)
        _, elapsed = _time_ms(slow)
        assert elapsed >= 40.0  # at least 40ms

    def test_propagates_exceptions(self) -> None:
        from tools.benchmark import _time_ms
        with pytest.raises(ValueError, match="boom"):
            _time_ms(lambda: (_ for _ in ()).throw(ValueError("boom")))


# ---------------------------------------------------------------------------
# _stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_min_avg_max_single_value(self) -> None:
        from tools.benchmark import _stats
        mn, avg, mx = _stats([100.0])
        assert mn == 100.0
        assert avg == 100.0
        assert mx == 100.0

    def test_min_correct(self) -> None:
        from tools.benchmark import _stats
        mn, _, _ = _stats([3.0, 1.0, 2.0])
        assert mn == 1.0

    def test_avg_correct(self) -> None:
        from tools.benchmark import _stats
        _, avg, _ = _stats([10.0, 20.0, 30.0])
        assert avg == 20.0

    def test_max_correct(self) -> None:
        from tools.benchmark import _stats
        _, _, mx = _stats([3.0, 1.0, 7.0, 2.0])
        assert mx == 7.0


# ---------------------------------------------------------------------------
# _format_table
# ---------------------------------------------------------------------------

class TestFormatTable:
    def test_output_is_string(self) -> None:
        from tools.benchmark import _format_table
        rows = [("Whisper transcribe", [100.0, 110.0, 90.0])]
        assert isinstance(_format_table(rows), str)

    def test_header_row_present(self) -> None:
        from tools.benchmark import _format_table
        out = _format_table([("Stage A", [10.0])])
        assert "min (ms)" in out
        assert "avg (ms)" in out
        assert "max (ms)" in out

    def test_separator_row_present(self) -> None:
        from tools.benchmark import _format_table
        out = _format_table([("Stage A", [10.0])])
        assert "---" in out

    def test_stage_name_appears_in_output(self) -> None:
        from tools.benchmark import _format_table
        out = _format_table([("Whisper transcribe", [50.0, 60.0])])
        assert "Whisper transcribe" in out

    def test_numeric_values_appear_in_output(self) -> None:
        from tools.benchmark import _format_table
        out = _format_table([("Stage", [100.0, 200.0, 150.0])])
        # min=100, avg=150, max=200
        assert "100" in out
        assert "200" in out

    def test_na_shown_for_none_timings(self) -> None:
        from tools.benchmark import _format_table
        out = _format_table([("Formatter (LLM)", None)])
        assert "N/A" in out

    def test_multiple_rows_all_present(self) -> None:
        from tools.benchmark import _format_table
        rows = [
            ("Whisper transcribe", [50.0]),
            ("Formatter (fast)", [1.0]),
            ("Formatter (LLM)", None),
            ("Text injection", [10.0]),
            ("TOTAL", [61.0]),
        ]
        out = _format_table(rows)
        for name, _ in rows:
            assert name in out

    def test_total_row_present(self) -> None:
        from tools.benchmark import _format_table
        rows = [("TOTAL", [100.0, 120.0, 110.0])]
        out = _format_table(rows)
        assert "TOTAL" in out


# ---------------------------------------------------------------------------
# _parse_args (benchmark CLI)
# ---------------------------------------------------------------------------

class TestBenchmarkParseArgs:
    def _parse(self, argv: list[str]):
        from tools.benchmark import _parse_args
        with patch("sys.argv", ["benchmark"] + argv):
            return _parse_args()

    def test_model_default_is_tiny_en(self) -> None:
        assert self._parse([]).model == "tiny.en"

    def test_model_override(self) -> None:
        assert self._parse(["--model", "base.en"]).model == "base.en"

    def test_backend_default_is_fast(self) -> None:
        assert self._parse([]).backend == "fast"

    def test_iterations_default_is_5(self) -> None:
        assert self._parse([]).iterations == 5

    def test_iterations_override(self) -> None:
        assert self._parse(["--iterations", "3"]).iterations == 3


# ---------------------------------------------------------------------------
# run_benchmark (mocking heavy deps)
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def test_returns_list_of_stage_tuples(self) -> None:
        from tools.benchmark import run_benchmark
        with (
            patch("tools.benchmark._transcribe", return_value="hello world"),
            patch("tools.benchmark._inject", return_value=True),
        ):
            rows = run_benchmark(
                wav_path=Path("tests/fixtures/sample.wav"),
                model="tiny.en",
                backend="fast",
                iterations=2,
            )
        assert isinstance(rows, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in rows)

    def test_whisper_row_has_timings(self) -> None:
        from tools.benchmark import run_benchmark
        with (
            patch("tools.benchmark._transcribe", return_value="test text"),
            patch("tools.benchmark._inject", return_value=True),
        ):
            rows = run_benchmark(
                wav_path=Path("tests/fixtures/sample.wav"),
                model="tiny.en",
                backend="fast",
                iterations=3,
            )
        whisper_row = next(r for r in rows if "Whisper" in r[0])
        assert whisper_row[1] is not None
        assert len(whisper_row[1]) == 3  # 3 iterations

    def test_fast_formatter_row_present(self) -> None:
        from tools.benchmark import run_benchmark
        with (
            patch("tools.benchmark._transcribe", return_value="some words here to format"),
            patch("tools.benchmark._inject", return_value=True),
        ):
            rows = run_benchmark(
                wav_path=Path("tests/fixtures/sample.wav"),
                model="tiny.en",
                backend="fast",
                iterations=2,
            )
        fast_row = next(r for r in rows if "fast" in r[0].lower())
        assert fast_row[1] is not None

    def test_llm_formatter_row_is_none_when_backend_fast(self) -> None:
        from tools.benchmark import run_benchmark
        with (
            patch("tools.benchmark._transcribe", return_value="text"),
            patch("tools.benchmark._inject", return_value=True),
        ):
            rows = run_benchmark(
                wav_path=Path("tests/fixtures/sample.wav"),
                model="tiny.en",
                backend="fast",
                iterations=2,
            )
        llm_row = next(r for r in rows if "LLM" in r[0])
        assert llm_row[1] is None  # skipped for fast backend

    def test_total_row_sums_available_stages(self) -> None:
        from tools.benchmark import run_benchmark
        with (
            patch("tools.benchmark._transcribe", return_value="text"),
            patch("tools.benchmark._inject", return_value=True),
        ):
            rows = run_benchmark(
                wav_path=Path("tests/fixtures/sample.wav"),
                model="tiny.en",
                backend="fast",
                iterations=2,
            )
        total_row = next(r for r in rows if r[0] == "TOTAL")
        assert total_row[1] is not None
        assert len(total_row[1]) == 2  # 2 iterations
