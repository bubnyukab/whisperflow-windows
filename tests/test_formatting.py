"""Unit tests for text formatters."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

from src.config.settings import Settings
from src.formatting import create_formatter
from src.formatting import format_text as router_format_text
from src.formatting.fast_formatter import FastFormatter, word_count
from src.formatting.ollama_formatter import OllamaFormatter


class TestFastFormatter:
    """Tests for the rule-based fast formatter."""

    def _fmt(self, text: str) -> str:
        return FastFormatter().format(text)

    def test_capitalizes_first_letter(self) -> None:
        assert self._fmt("hello world")[0] == "H"

    def test_adds_period_when_missing(self) -> None:
        assert self._fmt("hello world").endswith(".")

    def test_preserves_question_mark(self) -> None:
        result = self._fmt("is this working?")
        assert result.endswith("?")
        assert not result.endswith("?.")

    def test_preserves_exclamation(self) -> None:
        assert self._fmt("wow that works!").endswith("!")

    def test_collapses_whitespace(self) -> None:
        assert "  " not in self._fmt("hello   world")

    def test_empty_string(self) -> None:
        assert self._fmt("") == ""

    def test_returns_string(self) -> None:
        assert isinstance(self._fmt("test"), str)


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

    def test_payload_uses_completion_prompt_not_chat_format(self) -> None:
        """OllamaFormatter posts a 'prompt' completion, never chat 'system'/'messages'."""
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.post", return_value=self._ok_response("ok")) as mock_post:
            fmt.format("some text")
        body = mock_post.call_args.kwargs["json"]
        assert "prompt" in body
        assert "system" not in body
        assert "messages" not in body

    def test_prompt_contains_raw_transcript_in_completion_template(self) -> None:
        """The transcript is interpolated into the Input:/Output: completion template."""
        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.post", return_value=self._ok_response("ok")) as mock_post:
            fmt.format("send email to john")
        prompt = mock_post.call_args.kwargs["json"]["prompt"]
        assert "Input: send email to john" in prompt
        assert prompt.rstrip().endswith("Output:")

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
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Hello world."}
        mock_resp.raise_for_status = MagicMock()

        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.post", return_value=mock_resp):
            result = fmt.format("hello world")

        assert result == "Hello world."

    def test_connection_error_falls_back(self) -> None:
        import httpx as _httpx

        fmt = OllamaFormatter(url="http://localhost:11434", model="llama3.1:8b")
        with patch("httpx.post", side_effect=_httpx.ConnectError("refused")):
            result = fmt.format("hello world")

        assert result == "hello world"


class TestCreateFormatter:
    """Tests for the create_formatter router function."""

    def test_fast_backend_returns_fast_formatter(self) -> None:
        settings = Settings(formatter_backend="fast")
        assert isinstance(create_formatter(settings), FastFormatter)

    def test_ollama_backend_returns_ollama_formatter(self) -> None:
        settings = Settings(formatter_backend="ollama")
        fmt = create_formatter(settings)
        assert isinstance(fmt, OllamaFormatter)

    def test_ollama_formatter_uses_settings_url(self) -> None:
        settings = Settings(formatter_backend="ollama", ollama_url="http://myhost:11434")
        fmt = create_formatter(settings)
        assert fmt._url == "http://myhost:11434"

    def test_ollama_formatter_uses_settings_model(self) -> None:
        settings = Settings(formatter_backend="ollama", ollama_model="phi3:mini")
        fmt = create_formatter(settings)
        assert fmt._model == "phi3:mini"

    def test_ollama_formatter_uses_settings_timeout(self) -> None:
        settings = Settings(formatter_backend="ollama", ollama_timeout=5.0)
        fmt = create_formatter(settings)
        assert fmt._timeout == 5.0

class TestRouterFormatText:
    """Tests for the two-tier format_text router."""

    def _short_text(self, threshold: int = 10) -> str:
        return " ".join(["word"] * (threshold - 1))

    def _long_text(self, threshold: int = 10) -> str:
        return " ".join(["word"] * threshold)

    def test_short_input_returns_fast_result(self) -> None:
        settings = Settings(formatter_backend="ollama", llm_word_threshold=10)
        result = router_format_text(self._short_text(), settings)
        assert isinstance(result, str)
        assert result != ""

    def test_short_input_never_calls_llm(self) -> None:
        settings = Settings(formatter_backend="ollama", llm_word_threshold=10)
        with patch("src.formatting.OllamaFormatter") as mock_cls:
            router_format_text(self._short_text(), settings)
        mock_cls.return_value.format.assert_not_called()

    def test_fast_backend_never_calls_llm_regardless_of_length(self) -> None:
        settings = Settings(formatter_backend="fast", llm_word_threshold=10)
        with patch("src.formatting.OllamaFormatter") as mock_cls:
            router_format_text(self._long_text(), settings)
        mock_cls.return_value.format.assert_not_called()

    def test_long_input_ollama_backend_calls_llm(self) -> None:
        settings = Settings(formatter_backend="ollama", llm_word_threshold=10)
        mock_fmt = MagicMock()
        mock_fmt.format.return_value = "LLM result."
        with patch("src.formatting.create_formatter", return_value=mock_fmt):
            result = router_format_text(self._long_text(), settings)
        mock_fmt.format.assert_called_once()
        assert result == "LLM result."

    def test_llm_empty_response_falls_back_to_fast(self) -> None:
        settings = Settings(formatter_backend="ollama", llm_word_threshold=10)
        mock_fmt = MagicMock()
        mock_fmt.format.return_value = ""
        with patch("src.formatting.create_formatter", return_value=mock_fmt):
            result = router_format_text(self._long_text(), settings)
        assert result != ""

    def test_context_hint_passed_to_llm(self) -> None:
        settings = Settings(formatter_backend="ollama", llm_word_threshold=10)
        mock_fmt = MagicMock()
        mock_fmt.format.return_value = "ok"
        with patch("src.formatting.create_formatter", return_value=mock_fmt):
            router_format_text(self._long_text(), settings, context_hint="email")
        mock_fmt.format.assert_called_once_with(self._long_text(), "email")

    def test_exact_threshold_triggers_llm(self) -> None:
        settings = Settings(formatter_backend="ollama", llm_word_threshold=5)
        text = " ".join(["word"] * 5)
        mock_fmt = MagicMock()
        mock_fmt.format.return_value = "ok"
        with patch("src.formatting.create_formatter", return_value=mock_fmt):
            router_format_text(text, settings)
        mock_fmt.format.assert_called_once()

    def test_one_below_threshold_skips_llm(self) -> None:
        settings = Settings(formatter_backend="ollama", llm_word_threshold=5)
        text = " ".join(["word"] * 4)
        mock_fmt = MagicMock()
        with patch("src.formatting.create_formatter", return_value=mock_fmt):
            router_format_text(text, settings)
        mock_fmt.format.assert_not_called()
