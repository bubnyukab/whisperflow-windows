"""Unit tests for text formatters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config.settings import Settings
from src.formatting.fast_formatter import FastFormatterResult, format_text, word_count


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
