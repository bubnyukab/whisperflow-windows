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
