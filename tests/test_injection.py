"""Unit tests for TextInjector."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.injection.text_injector import TextInjector


class TestTextInjectorInject:
    """Tests for TextInjector.inject — mocks pyperclip, keyboard, time.sleep."""

    def _run(
        self,
        text: str = "hello",
        old_clipboard: str = "old text",
        paste_raises: bool = False,
        keyboard_raises: bool = False,
    ) -> tuple[bool, MagicMock, MagicMock, MagicMock]:
        """Helper: run inject with standard mocks, return (result, mock_paste, mock_copy, mock_send)."""
        mock_paste = MagicMock(
            side_effect=Exception("non-text") if paste_raises else None,
            return_value=old_clipboard,
        )
        mock_copy = MagicMock()
        mock_send = MagicMock(side_effect=Exception("kb error") if keyboard_raises else None)
        with (
            patch("src.injection.text_injector.pyperclip.paste", mock_paste),
            patch("src.injection.text_injector.pyperclip.copy", mock_copy),
            patch("src.injection.text_injector.keyboard.send", mock_send),
            patch("src.injection.text_injector.time.sleep"),
        ):
            result = TextInjector().inject(text)
        return result, mock_paste, mock_copy, mock_send

    def test_returns_true_on_success(self) -> None:
        result, *_ = self._run()
        assert result is True

    def test_copies_text_to_clipboard(self) -> None:
        _, _, mock_copy, _ = self._run(text="inject me")
        assert call("inject me") in mock_copy.call_args_list

    def test_sends_ctrl_v(self) -> None:
        _, _, _, mock_send = self._run()
        mock_send.assert_called_once_with("ctrl+v")

    def test_restores_old_clipboard_after_inject(self) -> None:
        _, _, mock_copy, _ = self._run(text="new", old_clipboard="saved")
        # last copy call must restore old clipboard
        assert mock_copy.call_args_list[-1] == call("saved")

    def test_order_of_operations(self) -> None:
        """copy(text) → sleep(0.05) → send(ctrl+v) → sleep(0.5) → copy(old)."""
        calls: list[str] = []
        with (
            patch("src.injection.text_injector.pyperclip.paste", return_value="old"),
            patch("src.injection.text_injector.pyperclip.copy", side_effect=lambda t: calls.append(f"copy({t})")),
            patch("src.injection.text_injector.keyboard.send", side_effect=lambda k: calls.append(f"send({k})")),
            patch("src.injection.text_injector.time.sleep", side_effect=lambda s: calls.append(f"sleep({s})")),
        ):
            TextInjector().inject("hello")
        assert calls == [
            "copy(hello)",
            "sleep(0.05)",
            "send(ctrl+v)",
            "sleep(0.5)",
            "copy(old)",
        ]

    def test_skips_restore_when_clipboard_was_non_text(self) -> None:
        _, _, mock_copy, _ = self._run(paste_raises=True)
        # only one copy call: setting the new text; no restore
        assert mock_copy.call_count == 1

    def test_returns_false_on_keyboard_error(self) -> None:
        result, *_ = self._run(keyboard_raises=True)
        assert result is False

    def test_returns_false_on_copy_error(self) -> None:
        with (
            patch("src.injection.text_injector.pyperclip.paste", return_value="old"),
            patch("src.injection.text_injector.pyperclip.copy", side_effect=Exception("copy fail")),
            patch("src.injection.text_injector.keyboard.send"),
            patch("src.injection.text_injector.time.sleep"),
        ):
            result = TextInjector().inject("text")
        assert result is False

    def test_sleep_50ms_before_paste(self) -> None:
        sleep_calls: list[float] = []
        with (
            patch("src.injection.text_injector.pyperclip.paste", return_value=""),
            patch("src.injection.text_injector.pyperclip.copy"),
            patch("src.injection.text_injector.keyboard.send"),
            patch("src.injection.text_injector.time.sleep", side_effect=sleep_calls.append),
        ):
            TextInjector().inject("hello")
        assert sleep_calls[0] == pytest.approx(0.05)

    def test_sleep_500ms_after_paste(self) -> None:
        sleep_calls: list[float] = []
        with (
            patch("src.injection.text_injector.pyperclip.paste", return_value=""),
            patch("src.injection.text_injector.pyperclip.copy"),
            patch("src.injection.text_injector.keyboard.send"),
            patch("src.injection.text_injector.time.sleep", side_effect=sleep_calls.append),
        ):
            TextInjector().inject("hello")
        assert sleep_calls[1] == pytest.approx(0.5)


class TestTextInjectorToast:
    """Tests for TextInjector.show_fallback_toast — mocks tkinter."""

    def _mock_tk(self) -> tuple[MagicMock, MagicMock, MagicMock]:
        """Return (mock_tk_module, mock_root, mock_top)."""
        mock_root = MagicMock()
        mock_root.winfo_screenwidth.return_value = 1920
        mock_root.winfo_screenheight.return_value = 1080
        mock_top = MagicMock()
        mock_tk_mod = MagicMock()
        mock_tk_mod.Tk.return_value = mock_root
        mock_tk_mod.Toplevel.return_value = mock_top
        mock_tk_mod.Text.return_value = MagicMock()
        mock_tk_mod.Button.return_value = MagicMock()
        mock_tk_mod.Frame.return_value = MagicMock()
        mock_tk_mod.Scrollbar.return_value = MagicMock()
        return mock_tk_mod, mock_root, mock_top

    def test_toast_sets_always_on_top(self) -> None:
        mock_tk_mod, mock_root, mock_top = self._mock_tk()
        with (
            patch.dict("sys.modules", {"tkinter": mock_tk_mod}),
            patch("src.injection.text_injector.pyperclip.copy"),
        ):
            TextInjector().show_fallback_toast("some text")
        mock_top.wm_attributes.assert_called_with("-topmost", True)

    def test_toast_is_overrideredirect(self) -> None:
        mock_tk_mod, mock_root, mock_top = self._mock_tk()
        with (
            patch.dict("sys.modules", {"tkinter": mock_tk_mod}),
            patch("src.injection.text_injector.pyperclip.copy"),
        ):
            TextInjector().show_fallback_toast("some text")
        mock_top.overrideredirect.assert_called_with(True)

    def test_toast_calls_mainloop(self) -> None:
        mock_tk_mod, mock_root, mock_top = self._mock_tk()
        with (
            patch.dict("sys.modules", {"tkinter": mock_tk_mod}),
            patch("src.injection.text_injector.pyperclip.copy"),
        ):
            TextInjector().show_fallback_toast("some text")
        mock_root.mainloop.assert_called_once()

    def test_toast_schedules_auto_close(self) -> None:
        mock_tk_mod, mock_root, mock_top = self._mock_tk()
        with (
            patch.dict("sys.modules", {"tkinter": mock_tk_mod}),
            patch("src.injection.text_injector.pyperclip.copy"),
        ):
            TextInjector().show_fallback_toast("some text")
        # after() must be called with 15000ms
        after_calls = mock_root.after.call_args_list
        assert any(c.args[0] == 15000 for c in after_calls)

    def test_toast_does_not_raise_on_error(self) -> None:
        """Toast must never propagate exceptions (non-critical UI)."""
        with patch.dict("sys.modules", {"tkinter": None}):
            TextInjector().show_fallback_toast("text")  # should not raise
