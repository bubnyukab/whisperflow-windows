"""Unit tests for text formatters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config.settings import Settings
from src.formatting.fast_formatter import FastFormatter, FastFormatterResult, format_text, word_count


class TestFastFormatter:
    """Tests for the rule-based fast formatter."""

    def test_capitalizes_first_letter(self) -> None:
        assert format_text("hello world").text[0] == "H"

    def test_adds_period_when_missing(self) -> None:
        assert format_text("hello world").text.endswith(".")

    def test_preserves_question_mark(self) -> None:
        result = format_text("is this working?")
        assert result.text.endswith("?")
        assert not result.text.endswith("?.")

    def test_preserves_exclamation(self) -> None:
        result = format_text("wow that works!")
        assert result.text.endswith("!")

    def test_collapses_whitespace(self) -> None:
        assert "  " not in format_text("hello   world").text

    def test_empty_string(self) -> None:
        assert format_text("").text == ""

    def test_latency_non_negative(self) -> None:
        assert format_text("hello world").latency_ms >= 0

    def test_returns_dataclass(self) -> None:
        assert isinstance(format_text("test"), FastFormatterResult)


class TestWordCount:
    def test_single_word(self) -> None:
        assert word_count("hello") == 1

    def test_multiple_words(self) -> None:
        assert word_count("hello world foo bar") == 4

    def test_empty(self) -> None:
        assert word_count("") == 0

    def test_whitespace_only(self) -> None:
        assert word_count("   ") == 0


class TestFastFormatterClass:
    """Tests for the FastFormatter class (filler-word stripping + rules)."""

    def test_strips_filler_at_start(self) -> None:
        assert FastFormatter().format("um send an email to john") == "Send an email to john."

    def test_strips_multiple_fillers(self) -> None:
        assert FastFormatter().format("so basically i think we should uh meet tomorrow") == "I think we should meet tomorrow."

    def test_preserves_existing_punctuation(self) -> None:
        assert FastFormatter().format("Hello, how are you?") == "Hello, how are you?"

    def test_empty_string(self) -> None:
        assert FastFormatter().format("") == ""

    def test_only_fillers_returns_empty_no_period(self) -> None:
        assert FastFormatter().format("um uh like") == ""

    def test_case_insensitive_stripping(self) -> None:
        assert FastFormatter().format("UM send it") == "Send it."

    def test_whole_word_matching_does_not_strip_prefix(self) -> None:
        # "like" should not clobber "likely"
        assert FastFormatter().format("a likely outcome") == "A likely outcome."

    def test_multi_word_filler_stripped(self) -> None:
        assert FastFormatter().format("you know it was great") == "It was great."

    def test_consecutive_fillers_yield_empty(self) -> None:
        assert FastFormatter().format("um uh") == ""

    def test_question_mark_preserved(self) -> None:
        assert FastFormatter().format("um is this working?") == "Is this working?"

    def test_exclamation_preserved(self) -> None:
        assert FastFormatter().format("uh wow that works!") == "Wow that works!"

    def test_capitalises_first_letter(self) -> None:
        assert FastFormatter().format("actually send the report")[0] == "S"

    def test_returns_str(self) -> None:
        assert isinstance(FastFormatter().format("hello"), str)

    def test_collapses_internal_whitespace(self) -> None:
        result = FastFormatter().format("so   hello   world")
        assert "  " not in result

    def test_latency_under_5ms(self) -> None:
        import time as _time
        ff = FastFormatter()
        t0 = _time.perf_counter()
        ff.format("um so basically this is a test you know right")
        assert (_time.perf_counter() - t0) * 1000 < 5.0


class TestOllamaFormatter:
    """Tests for the Ollama formatter with mocked httpx."""

    def test_success_path(self) -> None:
        from src.formatting.ollama_formatter import format_text

        settings = Settings()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Hello world."}
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp

        with patch("httpx.Client", return_value=mock_client):
            result = format_text("hello world", settings)

        assert result.success is True
        assert result.text == "Hello world."

    def test_connection_error_falls_back(self) -> None:
        from src.formatting.ollama_formatter import format_text
        import httpx

        settings = Settings()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("refused")

        with patch("httpx.Client", return_value=mock_client):
            result = format_text("hello world", settings)

        assert result.success is False
        assert result.text == "hello world"
