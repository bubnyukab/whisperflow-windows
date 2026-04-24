"""Unit tests for HotkeyListener."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.hotkey.listener import HotkeyListener, _to_pynput_format


def _make(
    hotkey: str = "ctrl+shift+r",
    on_press: object = None,
    on_release: object = None,
    mode: str = "hold",
) -> HotkeyListener:
    return HotkeyListener(
        hotkey=hotkey,
        on_press=on_press or (lambda: None),
        on_release=on_release or (lambda: None),
        mode=mode,
    )


@contextmanager
def _pynput_mock() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    """Patch pynput keyboard so start() succeeds without real system access."""
    with patch("src.hotkey.listener.keyboard") as mock_kb:
        mock_kb.HotKey.parse.return_value = frozenset()
        mock_kb.HotKey.return_value = MagicMock()
        mock_listener = MagicMock()
        mock_kb.Listener.return_value = mock_listener
        yield mock_kb, mock_listener


class TestHotkeyFormat:
    def test_modifier_wrapped_in_angle_brackets(self) -> None:
        assert _to_pynput_format("ctrl") == "<ctrl>"

    def test_special_key_wrapped_in_angle_brackets(self) -> None:
        assert _to_pynput_format("f9") == "<f9>"

    def test_single_alpha_stays_bare(self) -> None:
        assert _to_pynput_format("r") == "r"

    def test_combination_converted(self) -> None:
        assert _to_pynput_format("ctrl+shift+r") == "<ctrl>+<shift>+r"

    def test_ctrl_f9_converted(self) -> None:
        assert _to_pynput_format("ctrl+f9") == "<ctrl>+<f9>"

    def test_win_aliased_to_cmd(self) -> None:
        assert _to_pynput_format("win+r") == "<cmd>+r"

    def test_case_insensitive(self) -> None:
        assert _to_pynput_format("CTRL+F9") == "<ctrl>+<f9>"

    def test_space_key_wrapped(self) -> None:
        assert _to_pynput_format("space") == "<space>"


class TestLifecycle:
    def test_is_listening_false_before_start(self) -> None:
        assert _make().is_listening is False

    def test_is_listening_true_after_start(self) -> None:
        with _pynput_mock():
            listener = _make()
            listener.start()
            assert listener.is_listening is True
            listener.stop()

    def test_is_listening_false_after_stop(self) -> None:
        with _pynput_mock():
            listener = _make()
            listener.start()
            listener.stop()
            assert listener.is_listening is False

    def test_stop_before_start_is_safe(self) -> None:
        with _pynput_mock():
            _make().stop()  # must not raise

    def test_start_twice_registers_listener_only_once(self) -> None:
        with _pynput_mock() as (mock_kb, _):
            listener = _make(mode="hold")
            listener.start()
            listener.start()
            assert mock_kb.Listener.call_count == 1
            listener.stop()

    def test_stop_calls_underlying_listener_stop(self) -> None:
        with _pynput_mock() as (_, mock_listener):
            listener = _make()
            listener.start()
            listener.stop()
            mock_listener.stop.assert_called_once()


class TestHoldMode:
    def test_start_creates_one_listener(self) -> None:
        with _pynput_mock() as (mock_kb, _):
            listener = _make(mode="hold")
            listener.start()
            assert mock_kb.Listener.call_count == 1
            listener.stop()

    def test_press_callback_fires(self) -> None:
        pressed: list[int] = []
        with _pynput_mock():
            listener = _make(on_press=lambda: pressed.append(1), mode="hold")
            listener.start()
            listener._handle_press()
            listener.stop()
        assert pressed == [1]

    def test_release_callback_fires(self) -> None:
        released: list[int] = []
        with _pynput_mock():
            listener = _make(on_release=lambda: released.append(1), mode="hold")
            listener.start()
            listener._handle_release()
            listener.stop()
        assert released == [1]

    def test_press_and_release_fire_independently(self) -> None:
        events: list[str] = []
        with _pynput_mock():
            listener = _make(
                on_press=lambda: events.append("press"),
                on_release=lambda: events.append("release"),
                mode="hold",
            )
            listener.start()
            listener._handle_press()
            listener._handle_release()
            listener.stop()
        assert events == ["press", "release"]


class TestToggleMode:
    def test_start_creates_one_listener_for_toggle(self) -> None:
        with _pynput_mock() as (mock_kb, _):
            listener = _make(mode="toggle")
            listener.start()
            assert mock_kb.Listener.call_count == 1
            listener.stop()

    def test_first_keydown_fires_on_press(self) -> None:
        events: list[str] = []
        with _pynput_mock():
            listener = _make(
                on_press=lambda: events.append("press"),
                on_release=lambda: events.append("release"),
                mode="toggle",
            )
            listener.start()
            listener._handle_toggle()
            listener.stop()
        assert events == ["press"]

    def test_second_keydown_fires_on_release(self) -> None:
        events: list[str] = []
        with _pynput_mock():
            listener = _make(
                on_press=lambda: events.append("press"),
                on_release=lambda: events.append("release"),
                mode="toggle",
            )
            listener.start()
            with patch("src.hotkey.listener.time") as mock_time:
                mock_time.time.side_effect = [1.0, 1.2]  # 200ms apart > debounce
                listener._handle_toggle()
                listener._handle_toggle()
            listener.stop()
        assert events == ["press", "release"]

    def test_third_keydown_fires_on_press_again(self) -> None:
        events: list[str] = []
        with _pynput_mock():
            listener = _make(
                on_press=lambda: events.append("press"),
                on_release=lambda: events.append("release"),
                mode="toggle",
            )
            listener.start()
            with patch("src.hotkey.listener.time") as mock_time:
                mock_time.time.side_effect = [1.0, 1.2, 1.4]  # 200ms each > debounce
                listener._handle_toggle()
                listener._handle_toggle()
                listener._handle_toggle()
            listener.stop()
        assert events == ["press", "release", "press"]


class TestDebounce:
    """Debounce: re-fires within 150ms of previous event are ignored."""

    def test_rapid_refire_within_150ms_ignored_for_press(self) -> None:
        pressed: list[int] = []
        with _pynput_mock():
            listener = _make(on_press=lambda: pressed.append(1), mode="hold")
            listener.start()
            with patch("src.hotkey.listener.time") as mock_time:
                mock_time.time.side_effect = [1.0, 1.05, 1.1]
                listener._handle_press()  # t=1.0 → fires
                listener._handle_press()  # t=1.05 → 0.05 < 0.15, debounced
                listener._handle_press()  # t=1.1 → 0.1 < 0.15, debounced
            listener.stop()
        assert pressed == [1]

    def test_press_after_debounce_window_fires(self) -> None:
        pressed: list[int] = []
        with _pynput_mock():
            listener = _make(on_press=lambda: pressed.append(1), mode="hold")
            listener.start()
            with patch("src.hotkey.listener.time") as mock_time:
                mock_time.time.side_effect = [1.0, 1.2]
                listener._handle_press()  # t=1.0 → fires
                listener._handle_press()  # t=1.2 → 0.2 > 0.15, fires
            listener.stop()
        assert pressed == [1, 1]

    def test_rapid_refire_within_150ms_ignored_for_release(self) -> None:
        released: list[int] = []
        with _pynput_mock():
            listener = _make(on_release=lambda: released.append(1), mode="hold")
            listener.start()
            with patch("src.hotkey.listener.time") as mock_time:
                mock_time.time.side_effect = [1.0, 1.05]
                listener._handle_release()  # fires
                listener._handle_release()  # debounced
            listener.stop()
        assert released == [1]

    def test_toggle_debounce_ignores_rapid_second_keydown(self) -> None:
        events: list[str] = []
        with _pynput_mock():
            listener = _make(
                on_press=lambda: events.append("press"),
                on_release=lambda: events.append("release"),
                mode="toggle",
            )
            listener.start()
            with patch("src.hotkey.listener.time") as mock_time:
                mock_time.time.side_effect = [1.0, 1.05]
                listener._handle_toggle()  # t=1.0 → fires on_press
                listener._handle_toggle()  # t=1.05 → debounced, does NOT fire on_release
            listener.stop()
        assert events == ["press"]


class TestExceptionSafety:
    def test_on_press_exception_does_not_crash_listener(self) -> None:
        def bad_press() -> None:
            raise RuntimeError("boom")

        with _pynput_mock():
            listener = _make(on_press=bad_press, mode="hold")
            listener.start()
            listener._handle_press()  # must not raise
            assert listener.is_listening is True
            listener.stop()

    def test_on_release_exception_does_not_crash_listener(self) -> None:
        def bad_release() -> None:
            raise RuntimeError("boom")

        with _pynput_mock():
            listener = _make(on_release=bad_release, mode="hold")
            listener.start()
            listener._handle_release()  # must not raise
            assert listener.is_listening is True
            listener.stop()

    def test_toggle_exception_does_not_crash_listener(self) -> None:
        def bad_press() -> None:
            raise RuntimeError("boom")

        with _pynput_mock():
            listener = _make(on_press=bad_press, mode="toggle")
            listener.start()
            listener._handle_toggle()  # must not raise
            assert listener.is_listening is True
            listener.stop()

    def test_listener_stop_error_does_not_crash_stop(self) -> None:
        with _pynput_mock() as (_, mock_listener):
            mock_listener.stop.side_effect = Exception("stop failed")
            listener = _make()
            listener.start()
            listener.stop()  # must not raise
