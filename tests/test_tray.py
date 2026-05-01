"""Unit tests for src/tray/tray_app.py and src/tray/settings_ui.py."""

from __future__ import annotations

import threading
from typing import Callable
from unittest.mock import ANY, MagicMock, patch

import pytest
from PIL import Image

from src.config.settings import Settings
from src.tray.tray_app import TrayApp, make_circle_icon, _TRAINING_COLOR


# ---------------------------------------------------------------------------
# make_circle_icon — pure PIL, no mocks required
# ---------------------------------------------------------------------------

class TestMakeCircleIcon:
    def test_returns_pil_image(self) -> None:
        assert isinstance(make_circle_icon("idle"), Image.Image)

    def test_size_is_64x64_for_all_states(self) -> None:
        for state in ("idle", "recording", "processing", "done"):
            assert make_circle_icon(state).size == (64, 64), f"wrong size for state={state}"

    def test_idle_center_pixel_is_grey(self) -> None:
        img = make_circle_icon("idle")
        r, g, b, _ = img.getpixel((32, 32))
        assert r == g == b, f"Expected grey (R=G=B), got ({r},{g},{b})"

    def test_recording_center_pixel_has_dominant_red(self) -> None:
        img = make_circle_icon("recording")
        r, g, b, _ = img.getpixel((32, 32))
        assert r > g and r > b, f"Expected red dominant, got ({r},{g},{b})"

    def test_processing_center_pixel_has_dominant_blue(self) -> None:
        img = make_circle_icon("processing")
        r, g, b, _ = img.getpixel((32, 32))
        assert b > r and b > g, f"Expected blue dominant, got ({r},{g},{b})"

    def test_done_center_pixel_has_dominant_green(self) -> None:
        img = make_circle_icon("done")
        r, g, b, _ = img.getpixel((32, 32))
        assert g > r and g > b, f"Expected green dominant, got ({r},{g},{b})"

    def test_corners_are_transparent(self) -> None:
        img = make_circle_icon("idle")
        _, _, _, alpha = img.getpixel((0, 0))
        assert alpha == 0, "Corner pixel should be transparent (outside circle)"

    def test_image_mode_supports_transparency(self) -> None:
        img = make_circle_icon("idle")
        assert img.mode == "RGBA"

    def test_training_mode_false_no_orange_indicator(self) -> None:
        img = make_circle_icon("idle", training_mode=False)
        r, g, b, _ = img.getpixel((52, 12))  # top-right indicator area
        assert (r, g, b) != _TRAINING_COLOR, "Orange indicator should be absent"

    def test_training_mode_true_has_orange_indicator(self) -> None:
        img = make_circle_icon("idle", training_mode=True)
        r, g, b, a = img.getpixel((52, 12))  # center of orange dot region
        assert (r, g, b) == _TRAINING_COLOR and a == 255, "Orange indicator should be present"

    def test_training_mode_does_not_change_icon_size(self) -> None:
        assert make_circle_icon("idle", training_mode=True).size == (64, 64)


# ---------------------------------------------------------------------------
# TrayApp state machine — uses object.__new__ to skip full initialization
# ---------------------------------------------------------------------------

def _bare_app() -> TrayApp:
    """Create a TrayApp with only state-machine attributes initialized."""
    app = object.__new__(TrayApp)
    app._state = "idle"
    app._state_lock = threading.Lock()
    app._tray = None
    app._done_timer = None
    app._indicator = MagicMock()
    app._training_mode = False
    return app


class TestTrayAppStateMachine:
    def test_set_state_updates_to_recording(self) -> None:
        app = _bare_app()
        app.set_state("recording")
        assert app._state == "recording"

    def test_set_state_updates_to_processing(self) -> None:
        app = _bare_app()
        app.set_state("processing")
        assert app._state == "processing"

    def test_set_state_updates_to_done(self) -> None:
        app = _bare_app()
        with patch("src.tray.tray_app.threading.Timer"):
            app.set_state("done")
        assert app._state == "done"

    def test_done_state_schedules_15s_reset_timer(self) -> None:
        app = _bare_app()
        with patch("src.tray.tray_app.threading.Timer") as mock_cls:
            mock_timer = MagicMock()
            mock_cls.return_value = mock_timer
            app.set_state("done")
        mock_cls.assert_called_once_with(1.5, ANY)
        mock_timer.start.assert_called_once()

    def test_done_timer_is_daemon(self) -> None:
        app = _bare_app()
        with patch("src.tray.tray_app.threading.Timer") as mock_cls:
            mock_timer = MagicMock()
            mock_cls.return_value = mock_timer
            app.set_state("done")
        assert mock_timer.daemon is True

    def test_set_state_cancels_previous_done_timer(self) -> None:
        app = _bare_app()
        old_timer = MagicMock()
        app._done_timer = old_timer
        app.set_state("recording")
        old_timer.cancel.assert_called_once()
        assert app._done_timer is None

    def test_non_done_state_does_not_start_timer(self) -> None:
        app = _bare_app()
        with patch("src.tray.tray_app.threading.Timer") as mock_cls:
            app.set_state("recording")
        mock_cls.assert_not_called()

    def test_set_state_updates_pystray_icon_image(self) -> None:
        app = _bare_app()
        mock_tray = MagicMock()
        app._tray = mock_tray
        app.set_state("recording")
        assert isinstance(mock_tray.icon, Image.Image)

    def test_set_state_calls_update_menu_when_tray_active(self) -> None:
        app = _bare_app()
        mock_tray = MagicMock()
        app._tray = mock_tray
        app.set_state("processing")
        mock_tray.update_menu.assert_called_once()

    def test_set_state_does_not_raise_when_tray_is_none(self) -> None:
        app = _bare_app()
        app._tray = None
        app.set_state("recording")  # must not raise


# ---------------------------------------------------------------------------
# TrayApp._status_text — dynamic menu label content
# ---------------------------------------------------------------------------

class TestStatusText:
    def test_idle_contains_ready(self) -> None:
        app = _bare_app()
        assert "Ready" in app._status_text()

    def test_recording_contains_recording(self) -> None:
        app = _bare_app()
        app._state = "recording"
        assert "Recording" in app._status_text()

    def test_processing_contains_processing(self) -> None:
        app = _bare_app()
        app._state = "processing"
        assert "Processing" in app._status_text()

    def test_done_contains_done(self) -> None:
        app = _bare_app()
        app._state = "done"
        assert "Done" in app._status_text()


# ---------------------------------------------------------------------------
# SettingsWindow — instantiation smoke tests (full GUI deferred to Playwright)
# ---------------------------------------------------------------------------

class TestSettingsWindowInstantiation:
    def test_instantiation_does_not_raise(self) -> None:
        from src.tray.settings_ui import SettingsWindow
        win = SettingsWindow(settings=Settings(), on_save=lambda s: None)
        assert win is not None

    def test_stores_initial_settings(self) -> None:
        from src.tray.settings_ui import SettingsWindow
        s = Settings(hotkey="ctrl+r", formatter_backend="fast")
        win = SettingsWindow(settings=s, on_save=lambda _: None)
        assert win._settings.hotkey == "ctrl+r"
        assert win._settings.formatter_backend == "fast"

    def test_on_save_callback_is_stored(self) -> None:
        from src.tray.settings_ui import SettingsWindow
        cb: Callable = lambda s: None
        win = SettingsWindow(settings=Settings(), on_save=cb)
        assert win._on_save is cb


# ---------------------------------------------------------------------------
# Training mode toggle
# ---------------------------------------------------------------------------

class TestTrainingMode:
    def test_training_mode_defaults_false(self) -> None:
        app = _bare_app()
        assert app.training_mode is False

    def test_toggle_training_mode_enables(self) -> None:
        app = _bare_app()
        app._toggle_training_mode(None, None)
        assert app.training_mode is True

    def test_toggle_training_mode_disables_again(self) -> None:
        app = _bare_app()
        app._toggle_training_mode(None, None)
        app._toggle_training_mode(None, None)
        assert app.training_mode is False

    def test_toggle_updates_tray_icon(self) -> None:
        app = _bare_app()
        mock_tray = MagicMock()
        app._tray = mock_tray
        app._toggle_training_mode(None, None)
        assert isinstance(mock_tray.icon, Image.Image)

    def test_toggle_calls_update_menu(self) -> None:
        app = _bare_app()
        mock_tray = MagicMock()
        app._tray = mock_tray
        app._toggle_training_mode(None, None)
        mock_tray.update_menu.assert_called_once()

    def test_set_state_preserves_training_mode_in_icon(self) -> None:
        app = _bare_app()
        mock_tray = MagicMock()
        app._tray = mock_tray
        app._training_mode = True
        app.set_state("recording")
        icon = mock_tray.icon
        r, g, b, a = icon.getpixel((52, 12))
        assert (r, g, b) == _TRAINING_COLOR and a == 255
