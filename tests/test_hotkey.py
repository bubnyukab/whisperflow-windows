"""Unit tests for HotkeyListener."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.hotkey.listener import HotkeyListener


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


class TestLifecycle:
    def test_is_listening_false_before_start(self) -> None:
        assert _make().is_listening is False

    def test_is_listening_true_after_start(self) -> None:
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make()
            listener.start()
            assert listener.is_listening is True
            listener.stop()

    def test_is_listening_false_after_stop(self) -> None:
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make()
            listener.start()
            listener.stop()
        assert listener.is_listening is False

    def test_stop_before_start_is_safe(self) -> None:
        with patch("src.hotkey.listener.keyboard"):
            _make().stop()  # must not raise

    def test_start_twice_registers_hotkey_only_once(self) -> None:
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(mode="hold")
            listener.start()
            first_count = mock_kb.add_hotkey.call_count
            listener.start()
            assert mock_kb.add_hotkey.call_count == first_count
            listener.stop()

    def test_stop_calls_remove_hotkey_for_each_handle(self) -> None:
        handle1, handle2 = MagicMock(), MagicMock()
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.side_effect = [handle1, handle2]
            listener = _make(mode="hold")
            listener.start()
            listener.stop()
            assert mock_kb.remove_hotkey.call_count == 2
            mock_kb.remove_hotkey.assert_any_call(handle1)
            mock_kb.remove_hotkey.assert_any_call(handle2)


class TestHoldMode:
    def test_registers_two_hotkeys_for_hold_mode(self) -> None:
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(mode="hold")
            listener.start()
            assert mock_kb.add_hotkey.call_count == 2
            listener.stop()

    def test_second_hotkey_uses_trigger_on_release(self) -> None:
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(mode="hold")
            listener.start()
            # second add_hotkey call must include trigger_on_release=True
            second_call_kwargs = mock_kb.add_hotkey.call_args_list[1].kwargs
            assert second_call_kwargs.get("trigger_on_release") is True
            listener.stop()

    def test_press_callback_fires(self) -> None:
        pressed: list[int] = []
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(on_press=lambda: pressed.append(1), mode="hold")
            listener.start()
            listener._handle_press()
            listener.stop()
        assert pressed == [1]

    def test_release_callback_fires(self) -> None:
        released: list[int] = []
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(on_release=lambda: released.append(1), mode="hold")
            listener.start()
            listener._handle_release()
            listener.stop()
        assert released == [1]

    def test_press_and_release_fire_independently(self) -> None:
        log: list[str] = []
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(
                on_press=lambda: log.append("press"),
                on_release=lambda: log.append("release"),
                mode="hold",
            )
            listener.start()
            listener._handle_press()
            listener._handle_release()
            listener.stop()
        assert log == ["press", "release"]


class TestToggleMode:
    def test_registers_one_hotkey_for_toggle_mode(self) -> None:
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(mode="toggle")
            listener.start()
            assert mock_kb.add_hotkey.call_count == 1
            listener.stop()

    def test_first_keydown_fires_on_press(self) -> None:
        log: list[str] = []
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(
                on_press=lambda: log.append("press"),
                on_release=lambda: log.append("release"),
                mode="toggle",
            )
            listener.start()
            listener._handle_toggle()
            listener.stop()
        assert log == ["press"]

    def test_second_keydown_fires_on_release(self) -> None:
        log: list[str] = []
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(
                on_press=lambda: log.append("press"),
                on_release=lambda: log.append("release"),
                mode="toggle",
            )
            listener.start()
            with patch("src.hotkey.listener.time") as mock_time:
                mock_time.time.side_effect = [1.0, 1.2]  # 200ms apart > debounce
                listener._handle_toggle()
                listener._handle_toggle()
            listener.stop()
        assert log == ["press", "release"]

    def test_third_keydown_fires_on_press_again(self) -> None:
        log: list[str] = []
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(
                on_press=lambda: log.append("press"),
                on_release=lambda: log.append("release"),
                mode="toggle",
            )
            listener.start()
            with patch("src.hotkey.listener.time") as mock_time:
                mock_time.time.side_effect = [1.0, 1.2, 1.4]  # 200ms each > debounce
                listener._handle_toggle()
                listener._handle_toggle()
                listener._handle_toggle()
            listener.stop()
        assert log == ["press", "release", "press"]


class TestDebounce:
    """Debounce: re-fires within 150ms of previous event are ignored."""

    def _simulate_press(self, listener: HotkeyListener, times: list[float]) -> None:
        with patch("src.hotkey.listener.time") as mock_time:
            mock_time.time.side_effect = iter(times)
            for _ in times:
                listener._handle_press()

    def test_rapid_refire_within_150ms_ignored_for_press(self) -> None:
        pressed: list[int] = []
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
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
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
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
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(on_release=lambda: released.append(1), mode="hold")
            listener.start()
            with patch("src.hotkey.listener.time") as mock_time:
                mock_time.time.side_effect = [1.0, 1.05]
                listener._handle_release()  # fires
                listener._handle_release()  # debounced
            listener.stop()
        assert released == [1]

    def test_toggle_debounce_ignores_rapid_second_keydown(self) -> None:
        log: list[str] = []
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(
                on_press=lambda: log.append("press"),
                on_release=lambda: log.append("release"),
                mode="toggle",
            )
            listener.start()
            with patch("src.hotkey.listener.time") as mock_time:
                mock_time.time.side_effect = [1.0, 1.05]
                listener._handle_toggle()  # t=1.0 → fires on_press
                listener._handle_toggle()  # t=1.05 → debounced, does NOT fire on_release
            listener.stop()
        assert log == ["press"]


class TestExceptionSafety:
    def test_on_press_exception_does_not_crash_listener(self) -> None:
        def bad_press() -> None:
            raise RuntimeError("boom")

        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(on_press=bad_press, mode="hold")
            listener.start()
            listener._handle_press()  # must not raise
            assert listener.is_listening is True
            listener.stop()

    def test_on_release_exception_does_not_crash_listener(self) -> None:
        def bad_release() -> None:
            raise RuntimeError("boom")

        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(on_release=bad_release, mode="hold")
            listener.start()
            listener._handle_release()  # must not raise
            assert listener.is_listening is True
            listener.stop()

    def test_toggle_exception_does_not_crash_listener(self) -> None:
        def bad_press() -> None:
            raise RuntimeError("boom")

        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            listener = _make(on_press=bad_press, mode="toggle")
            listener.start()
            listener._handle_toggle()  # must not raise
            assert listener.is_listening is True
            listener.stop()

    def test_remove_hotkey_error_does_not_crash_stop(self) -> None:
        with patch("src.hotkey.listener.keyboard") as mock_kb:
            mock_kb.add_hotkey.return_value = MagicMock()
            mock_kb.remove_hotkey.side_effect = Exception("unhook failed")
            listener = _make()
            listener.start()
            listener.stop()  # must not raise
