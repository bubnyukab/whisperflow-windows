"""Unit tests for text formatters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config.settings import Settings
from src.formatting.fast_formatter import FastFormatter, FastFormatterResult, format_text, word_count
from src.formatting.ollama_formatter import OllamaFormatter


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


class TestOllamaFormatterClass:
    """Tests for the OllamaFormatter class (mocked httpx.post)."""

    def _ok_response(self, text: str) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = {"response": text, "done": True}
        resp.raise_for_status = MagicMock()
        return resp

    def test_happy_path_returns_formatted_text(self) -> None:
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.post", return_value=self._ok_response("Send an email to John.")):
            assert fmt.format("send email to john") == "Send an email to John."

    def test_timeout_returns_raw_transcript(self) -> None:
        import httpx as _httpx
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.post", side_effect=_httpx.TimeoutException("timeout")):
            assert fmt.format("send email to john") == "send email to john"

    def test_connection_error_returns_raw_transcript(self) -> None:
        import httpx as _httpx
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.post", side_effect=_httpx.ConnectError("refused")):
            assert fmt.format("send email to john") == "send email to john"

    def test_unexpected_exception_returns_raw_transcript(self) -> None:
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.post", side_effect=RuntimeError("oops")):
            assert fmt.format("raw text") == "raw text"

    def test_context_hint_appended_to_system_prompt(self) -> None:
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.post", return_value=self._ok_response("ok")) as mock_post:
            fmt.format("some text", context_hint="composing an email")
        body = mock_post.call_args.kwargs["json"]
        assert "composing an email" in body["system"]

    def test_no_context_hint_system_prompt_unchanged(self) -> None:
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.post", return_value=self._ok_response("ok")) as mock_post:
            fmt.format("some text")
        body = mock_post.call_args.kwargs["json"]
        assert "Context:" not in body["system"]

    def test_posts_to_api_generate_endpoint(self) -> None:
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.post", return_value=self._ok_response("ok")) as mock_post:
            fmt.format("test")
        assert mock_post.call_args.args[0] == "http://localhost:11434/api/generate"

    def test_stream_is_false_in_payload(self) -> None:
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.post", return_value=self._ok_response("ok")) as mock_post:
            fmt.format("test")
        assert mock_post.call_args.kwargs["json"]["stream"] is False

    def test_configured_timeout_passed_to_httpx(self) -> None:
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b", timeout=5.0)
        with patch("httpx.post", return_value=self._ok_response("ok")) as mock_post:
            fmt.format("test")
        assert mock_post.call_args.kwargs["timeout"] == 5.0

    def test_strips_whitespace_from_response(self) -> None:
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.post", return_value=self._ok_response("  Formatted.  \n")):
            assert fmt.format("raw") == "Formatted."

    def test_is_available_true_on_200(self) -> None:
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.get", return_value=MagicMock(status_code=200)):
            assert fmt.is_available() is True

    def test_is_available_false_on_connect_error(self) -> None:
        import httpx as _httpx
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.get", side_effect=_httpx.ConnectError("refused")):
            assert fmt.is_available() is False

    def test_is_available_false_on_non_200(self) -> None:
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.get", return_value=MagicMock(status_code=503)):
            assert fmt.is_available() is False


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
