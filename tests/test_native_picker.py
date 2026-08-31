"""Native folder picker tests without opening a real GUI."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread, get_ident

import pytest

from coding_agent.web.native_picker import (
    NativeFolderPickerBusyError,
    NativeFolderPickerUnavailableError,
    WindowsNativeFolderPicker,
)


def test_picker_runs_dialog_in_a_dedicated_thread_and_returns_a_directory(
    tmp_path: Path,
) -> None:
    dialog_threads: list[int] = []

    def choose() -> str:
        dialog_threads.append(get_ident())
        return str(tmp_path)

    caller_thread = get_ident()
    picker = WindowsNativeFolderPicker(
        _platform_supported=True,
        _dialog_runner=choose,
    )

    assert picker.available is True
    assert picker.pick_directory() == tmp_path.resolve()
    assert dialog_threads and dialog_threads[0] != caller_thread


def test_picker_cancellation_has_no_path_side_effect(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())
    picker = WindowsNativeFolderPicker(
        _platform_supported=True,
        _dialog_runner=lambda: None,
    )

    assert picker.pick_directory() is None
    assert set(tmp_path.iterdir()) == before


def test_picker_rejects_reentry_without_waiting(tmp_path: Path) -> None:
    entered = Event()
    release = Event()
    first_result: list[Path | None] = []

    def choose() -> str:
        entered.set()
        assert release.wait(timeout=2)
        return str(tmp_path)

    picker = WindowsNativeFolderPicker(
        _platform_supported=True,
        _dialog_runner=choose,
    )
    first = Thread(target=lambda: first_result.append(picker.pick_directory()))
    first.start()
    assert entered.wait(timeout=1)

    with pytest.raises(NativeFolderPickerBusyError, match="已有文件夹选择窗口"):
        picker.pick_directory()

    release.set()
    first.join(timeout=2)
    assert not first.is_alive()
    assert first_result == [tmp_path.resolve()]


def test_picker_stabilizes_dialog_and_path_failures(tmp_path: Path) -> None:
    private_detail = f"PRIVATE: {tmp_path}"

    def fail() -> str:
        raise RuntimeError(private_detail)

    picker = WindowsNativeFolderPicker(
        _platform_supported=True,
        _dialog_runner=fail,
    )

    with pytest.raises(NativeFolderPickerUnavailableError) as raised:
        picker.pick_directory()

    assert str(raised.value) == "本机文件夹选择器暂时不可用"
    assert private_detail not in str(raised.value)
    assert picker.available is False
    with pytest.raises(NativeFolderPickerUnavailableError):
        picker.pick_directory()


def test_picker_is_unavailable_off_windows_and_when_tk_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsupported = WindowsNativeFolderPicker(_platform_supported=False)
    monkeypatch.setattr(
        "coding_agent.web.native_picker.importlib.util.find_spec",
        lambda name: None,
    )
    missing_tk = WindowsNativeFolderPicker(_platform_supported=True)

    assert unsupported.available is False
    assert missing_tk.available is False
    with pytest.raises(NativeFolderPickerUnavailableError):
        unsupported.pick_directory()
