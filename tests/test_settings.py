"""Unit tests for src/config/settings.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings, check_ollama, load_settings, save_settings


class TestSettingsDefaults:
    def test_whisper_model(self) -> None:
        assert Settings().whisper_model == "tiny.en"

    def test_hotkey(self) -> None:
        assert Settings().hotkey == "win+shift+space"

    def test_formatter_backend(self) -> None:
        assert Settings().formatter_backend == "ollama"

    def test_ollama_model(self) -> None:
        assert Settings().ollama_model == "llama3.1:8b"

    def test_ollama_url(self) -> None:
        assert Settings().ollama_url == "http://localhost:11434"

    def test_ollama_timeout(self) -> None:
        assert Settings().ollama_timeout == 15.0

    def test_llm_word_threshold(self) -> None:
        assert Settings().llm_word_threshold == 10

    def test_vad_silence_ms(self) -> None:
        assert Settings().vad_silence_ms == 400

    def test_language(self) -> None:
        assert Settings().language == "en"

    def test_recording_mode(self) -> None:
        assert Settings().recording_mode == "hold"

    def test_history_max(self) -> None:
        assert Settings().history_max == 50

    def test_anthropic_api_key_default_empty(self) -> None:
        assert Settings().anthropic_api_key == ""

    def test_models_dir_is_path(self) -> None:
        assert isinstance(Settings().models_dir, Path)

    def test_models_dir_default(self) -> None:
        assert Settings().models_dir == Path.home() / ".whisperflow" / "models"

    def test_log_dir_is_path(self) -> None:
        assert isinstance(Settings().log_dir, Path)

    def test_log_dir_default(self) -> None:
        assert Settings().log_dir == Path.home() / ".whisperflow" / "logs"


class TestLoadSettings:
    def test_returns_defaults_when_no_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", tmp_path / "config.json")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr("src.config.settings.load_dotenv", lambda **kw: None)
        s = load_settings()
        assert s.whisper_model == "tiny.en"
        assert s.llm_word_threshold == 10

    def test_creates_config_file_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        monkeypatch.setattr("src.config.settings.load_dotenv", lambda **kw: None)
        load_settings()
        assert config_path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "subdir" / "config.json"
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        monkeypatch.setattr("src.config.settings.load_dotenv", lambda **kw: None)
        load_settings()
        assert config_path.parent.exists()

    def test_reads_values_from_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"whisper_model": "base.en", "llm_word_threshold": 5}))
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        monkeypatch.setattr("src.config.settings.load_dotenv", lambda **kw: None)
        s = load_settings()
        assert s.whisper_model == "base.en"
        assert s.llm_word_threshold == 5

    def test_ignores_unknown_keys_in_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"unknown_key": "value", "whisper_model": "base.en"}))
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        monkeypatch.setattr("src.config.settings.load_dotenv", lambda **kw: None)
        s = load_settings()
        assert s.whisper_model == "base.en"

    def test_returns_defaults_on_invalid_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text("{{not valid json")
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        monkeypatch.setattr("src.config.settings.load_dotenv", lambda **kw: None)
        s = load_settings()
        assert s.whisper_model == "tiny.en"

    def test_api_key_from_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        monkeypatch.setattr("src.config.settings.load_dotenv", lambda **kw: None)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        s = load_settings()
        assert s.anthropic_api_key == "sk-test-key"

    def test_api_key_not_loaded_from_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"anthropic_api_key": "should-not-appear"}))
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        monkeypatch.setattr("src.config.settings.load_dotenv", lambda **kw: None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        s = load_settings()
        assert s.anthropic_api_key == ""

    def test_models_dir_loaded_as_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"models_dir": "/some/custom/path"}))
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        monkeypatch.setattr("src.config.settings.load_dotenv", lambda **kw: None)
        s = load_settings()
        assert isinstance(s.models_dir, Path)
        assert s.models_dir == Path("/some/custom/path")

    def test_log_dir_loaded_as_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"log_dir": "/logs/custom"}))
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        monkeypatch.setattr("src.config.settings.load_dotenv", lambda **kw: None)
        s = load_settings()
        assert isinstance(s.log_dir, Path)
        assert s.log_dir == Path("/logs/custom")


class TestSaveSettings:
    def test_writes_to_config_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        save_settings(Settings(whisper_model="medium.en"))
        data = json.loads(config_path.read_text())
        assert data["whisper_model"] == "medium.en"

    def test_excludes_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        save_settings(Settings(anthropic_api_key="sk-secret"))
        data = json.loads(config_path.read_text())
        assert "anthropic_api_key" not in data

    def test_creates_parent_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "subdir" / "config.json"
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        save_settings(Settings())
        assert config_path.exists()

    def test_path_fields_serialized_as_strings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        save_settings(Settings())
        data = json.loads(config_path.read_text())
        assert isinstance(data["models_dir"], str)
        assert isinstance(data["log_dir"], str)

    def test_round_trip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("src.config.settings.SETTINGS_PATH", config_path)
        monkeypatch.setattr("src.config.settings.load_dotenv", lambda **kw: None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        original = Settings(whisper_model="base.en", llm_word_threshold=20, history_max=100)
        save_settings(original)
        loaded = load_settings()
        assert loaded.whisper_model == "base.en"
        assert loaded.llm_word_threshold == 20
        assert loaded.history_max == 100


class TestCheckOllama:
    def test_returns_true_when_reachable(self) -> None:
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert check_ollama("http://localhost:11434") is True

    def test_calls_api_tags_endpoint(self) -> None:
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            check_ollama("http://localhost:11434")
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=2.0)

    def test_returns_false_on_connect_error(self) -> None:
        import httpx as _httpx
        with patch("httpx.get", side_effect=_httpx.ConnectError("refused")):
            assert check_ollama("http://localhost:11434") is False

    def test_returns_false_on_timeout(self) -> None:
        import httpx as _httpx
        with patch("httpx.get", side_effect=_httpx.TimeoutException("timeout")):
            assert check_ollama("http://localhost:11434") is False

    def test_never_raises_on_unexpected_exception(self) -> None:
        with patch("httpx.get", side_effect=RuntimeError("unexpected")):
            assert check_ollama("http://localhost:11434") is False

    def test_uses_two_second_timeout(self) -> None:
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            check_ollama("http://custom:9999")
        _, kwargs = mock_get.call_args
        assert kwargs.get("timeout") == 2.0
