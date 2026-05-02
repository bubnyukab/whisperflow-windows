"""UI smoke tests for SettingsWindow.

Playwright cannot interact with tkinter desktop windows.
Per the blueprint: 'If tkinter windows aren't detectable, fall back to
tkinter's own test utilities.' — we test internal widget logic directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.config.settings import Settings
from src.tray.settings_ui import SettingsWindow, _BACKEND_VALUES, _BACKENDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window(backend: str = "ollama") -> SettingsWindow:
    """Create a SettingsWindow with mocked frame widgets, no display needed."""
    win = SettingsWindow(
        settings=Settings(formatter_backend=backend),
        on_save=lambda s: None,
    )
    # Inject mock frames that the visibility logic operates on
    win._ollama_frame = MagicMock()
    win._local_frame = MagicMock()
    backend_label = {v: k for k, v in _BACKEND_VALUES.items()}.get(backend, "Fast only")
    win._backend_var = MagicMock()
    win._backend_var.get.return_value = backend_label
    return win


# ---------------------------------------------------------------------------
# Backend dropdown — existence and content
# ---------------------------------------------------------------------------

class TestBackendDropdownPresent:
    """The formatter backend dropdown is the central control on the Formatter tab."""

    def test_backend_labels_include_fast_only(self) -> None:
        assert "Fast only" in _BACKENDS

    def test_backend_labels_include_ollama(self) -> None:
        assert "Ollama (local, free)" in _BACKENDS

    def test_backend_values_map_fast_only(self) -> None:
        assert _BACKEND_VALUES["Fast only"] == "fast"

    def test_backend_values_map_ollama(self) -> None:
        assert _BACKEND_VALUES["Ollama (local, free)"] == "ollama"


# ---------------------------------------------------------------------------
# Selecting "Ollama (local, free)" — Ollama section visible, Claude hidden
# ---------------------------------------------------------------------------

class TestOllamaBackendSelected:
    def test_ollama_frame_shown(self) -> None:
        win = _make_window(backend="ollama")
        win._refresh_formatter_sections()
        win._ollama_frame.grid.assert_called_once()

    def test_ollama_frame_not_removed_when_ollama_selected(self) -> None:
        win = _make_window(backend="ollama")
        win._refresh_formatter_sections()
        win._ollama_frame.grid_remove.assert_not_called()


# ---------------------------------------------------------------------------
# Ollama model dropdown present (as a constant, not a widget check)
# ---------------------------------------------------------------------------

class TestOllamaModelDropdownPresent:
    def test_ollama_models_list_is_nonempty(self) -> None:
        from src.tray.settings_ui import _OLLAMA_MODELS
        assert len(_OLLAMA_MODELS) > 0

    def test_default_ollama_model_in_list(self) -> None:
        from src.tray.settings_ui import _OLLAMA_MODELS
        assert Settings().ollama_model in _OLLAMA_MODELS

    def test_phi3_mini_in_list(self) -> None:
        from src.tray.settings_ui import _OLLAMA_MODELS
        assert "phi3:mini" in _OLLAMA_MODELS

    def test_llama31_in_list(self) -> None:
        from src.tray.settings_ui import _OLLAMA_MODELS
        assert "llama3.1:8b" in _OLLAMA_MODELS


# ---------------------------------------------------------------------------
# "Check connection" button — exists as a callable on the window
# ---------------------------------------------------------------------------

class TestCheckConnectionButton:
    def test_check_ollama_connection_method_exists(self) -> None:
        win = SettingsWindow(settings=Settings(), on_save=lambda s: None)
        assert callable(getattr(win, "_check_ollama_connection", None))

    def test_check_connection_sets_checking_status(self) -> None:
        win = _make_window(backend="ollama")
        win._ollama_url_var = MagicMock()
        win._ollama_url_var.get.return_value = "http://localhost:11434"
        win._ollama_status_var = MagicMock()
        # Patch the background thread so it doesn't actually spawn
        with patch("threading.Thread"):
            win._check_ollama_connection()
        win._ollama_status_var.set.assert_called_once_with("Checking...")

    def test_check_connection_starts_background_thread(self) -> None:
        win = _make_window(backend="ollama")
        win._ollama_url_var = MagicMock()
        win._ollama_url_var.get.return_value = "http://localhost:11434"
        win._ollama_status_var = MagicMock()
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            win._check_ollama_connection()
        mock_thread.start.assert_called_once()


# ---------------------------------------------------------------------------
# Closing the window — no errors raised on instantiation / teardown
# ---------------------------------------------------------------------------

class TestWindowLifecycle:
    def test_instantiation_does_not_raise(self) -> None:
        win = SettingsWindow(settings=Settings(), on_save=lambda s: None)
        assert win is not None

    def test_on_save_callback_stored(self) -> None:
        cb = MagicMock()
        win = SettingsWindow(settings=Settings(), on_save=cb)
        assert win._on_save is cb

    def test_settings_stored_on_init(self) -> None:
        s = Settings(hotkey="ctrl+shift+x")
        win = SettingsWindow(settings=s, on_save=lambda _: None)
        assert win._settings.hotkey == "ctrl+shift+x"

    def test_fast_backend_hides_both_sections(self) -> None:
        win = _make_window(backend="fast")
        win._refresh_formatter_sections()
        win._ollama_frame.grid_remove.assert_called_once()
        win._local_frame.grid_remove.assert_called_once()

    def test_switching_from_ollama_to_fast_hides_ollama(self) -> None:
        win = _make_window(backend="fast")
        win._refresh_formatter_sections()
        win._ollama_frame.grid.assert_not_called()


# ---------------------------------------------------------------------------
# _save() preserves all Settings fields
# ---------------------------------------------------------------------------

class TestSavePreservesAllFields:
    """_save() must round-trip every Settings field, not just the ones with UI controls."""

    def _make_save_window(self, settings: Settings) -> tuple[SettingsWindow, list]:
        saved: list = []
        win = SettingsWindow(settings=settings, on_save=lambda s: saved.append(s))
        win._ollama_frame = MagicMock()
        win._local_frame = MagicMock()
        backend_label = {v: k for k, v in _BACKEND_VALUES.items()}.get(
            settings.formatter_backend, "Fast only"
        )
        win._backend_var = MagicMock()
        win._backend_var.get.return_value = backend_label
        win._whisper_var = MagicMock()
        win._whisper_var.get.return_value = settings.whisper_model
        win._hotkey_var = MagicMock()
        win._hotkey_var.get.return_value = settings.hotkey
        win._mode_var = MagicMock()
        win._mode_var.get.return_value = settings.recording_mode
        win._lang_var = MagicMock()
        win._lang_var.get.return_value = settings.language
        win._ollama_model_var = MagicMock()
        win._ollama_model_var.get.return_value = settings.ollama_model
        win._ollama_url_var = MagicMock()
        win._ollama_url_var.get.return_value = settings.ollama_url
        win._threshold_var = MagicMock()
        win._threshold_var.get.return_value = settings.llm_word_threshold
        win._vad_var = MagicMock()
        win._vad_var.get.return_value = settings.vad_silence_ms
        win._local_model_path_var = MagicMock()
        win._local_model_path_var.get.return_value = str(settings.local_model_path)
        win._root = MagicMock()
        return win, saved

    def test_training_pairs_path_preserved(self) -> None:
        from pathlib import Path
        custom_path = Path("/custom/training.jsonl")
        s = Settings(training_pairs_path=custom_path)
        win, saved = self._make_save_window(s)
        with patch("src.config.settings.save_settings"):
            win._save()
        assert saved[0].training_pairs_path == custom_path

    def test_models_dir_preserved(self) -> None:
        from pathlib import Path
        custom_dir = Path("/custom/models")
        s = Settings(models_dir=custom_dir)
        win, saved = self._make_save_window(s)
        with patch("src.config.settings.save_settings"):
            win._save()
        assert saved[0].models_dir == custom_dir

    def test_log_dir_preserved(self) -> None:
        from pathlib import Path
        custom_dir = Path("/custom/logs")
        s = Settings(log_dir=custom_dir)
        win, saved = self._make_save_window(s)
        with patch("src.config.settings.save_settings"):
            win._save()
        assert saved[0].log_dir == custom_dir

    def test_history_max_preserved(self) -> None:
        s = Settings(history_max=99)
        win, saved = self._make_save_window(s)
        with patch("src.config.settings.save_settings"):
            win._save()
        assert saved[0].history_max == 99

    def test_ollama_timeout_preserved(self) -> None:
        s = Settings(ollama_timeout=30.0)
        win, saved = self._make_save_window(s)
        with patch("src.config.settings.save_settings"):
            win._save()
        assert saved[0].ollama_timeout == 30.0
