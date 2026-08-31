"""Small, optional bridge to the native Windows folder chooser.

The Web API depends only on :class:`NativeFolderPicker`, which keeps tests and
non-Windows hosts independent from Tk.  The concrete picker lazily imports
``tkinter`` and owns the complete Tk lifecycle inside one short-lived thread.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path
from threading import Lock, Thread
from typing import Protocol

_PICKER_BUSY_MESSAGE = "已有文件夹选择窗口正在等待操作"
_PICKER_UNAVAILABLE_MESSAGE = "本机文件夹选择器暂时不可用"


class NativeFolderPickerError(RuntimeError):
    """Base class for stable, path-free picker failures."""


class NativeFolderPickerBusyError(NativeFolderPickerError):
    """Another native picker dialog is already active."""


class NativeFolderPickerUnavailableError(NativeFolderPickerError):
    """The platform or GUI runtime cannot provide a native picker."""


class NativeFolderPicker(Protocol):
    """Narrow injectable surface used by the local Web host."""

    @property
    def available(self) -> bool: ...

    def pick_directory(self) -> Path | None: ...


class WindowsNativeFolderPicker:
    """Open one Windows folder dialog without making Tk a runtime dependency.

    A non-blocking process-local lock prevents multiple dialogs.  The dialog
    thread catches every ordinary GUI/path failure and reports only a stable
    domain error to callers; raw exception text and partially selected paths
    never cross the API boundary.
    """

    def __init__(
        self,
        *,
        _platform_supported: bool | None = None,
        _dialog_runner: Callable[[], str | None] | None = None,
    ) -> None:
        platform_supported = (
            sys.platform == "win32" and os.name == "nt"
            if _platform_supported is None
            else _platform_supported
        )
        self._dialog_runner = _dialog_runner or _show_tk_directory_dialog
        self._available = platform_supported and (
            _dialog_runner is not None or _tkinter_is_importable()
        )
        self._dialog_lock = Lock()

    @property
    def available(self) -> bool:
        return self._available

    def pick_directory(self) -> Path | None:
        if not self._dialog_lock.acquire(blocking=False):
            raise NativeFolderPickerBusyError(_PICKER_BUSY_MESSAGE)
        if not self._available:
            self._dialog_lock.release()
            raise NativeFolderPickerUnavailableError(_PICKER_UNAVAILABLE_MESSAGE)

        selected: list[Path | None] = []
        failed: list[bool] = []

        def run_dialog() -> None:
            try:
                raw_path = self._dialog_runner()
                if not raw_path:
                    selected.append(None)
                    return
                resolved = Path(raw_path).resolve(strict=True)
                if not resolved.is_dir():
                    failed.append(True)
                    return
                selected.append(resolved)
            except Exception:
                failed.append(True)

        try:
            dialog_thread = Thread(
                target=run_dialog,
                name="coding-agent-folder-picker",
                daemon=True,
            )
            dialog_thread.start()
            dialog_thread.join()
        except Exception:
            failed.append(True)
        finally:
            if failed or len(selected) != 1:
                # Publish runtime unavailability while the lock is still held, so
                # the next caller cannot observe a stale available=True value.
                self._available = False
            self._dialog_lock.release()

        if failed or len(selected) != 1:
            raise NativeFolderPickerUnavailableError(_PICKER_UNAVAILABLE_MESSAGE)
        return selected[0]


def _tkinter_is_importable() -> bool:
    try:
        return importlib.util.find_spec("tkinter") is not None
    except (ImportError, ValueError):
        return False


def _show_tk_directory_dialog() -> str | None:
    """Create, use, and destroy Tk in the same dedicated picker thread."""

    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        root.update_idletasks()
        selected = filedialog.askdirectory(
            parent=root,
            title="选择项目文件夹",
            mustexist=True,
        )
        return str(selected) if selected else None
    finally:
        root.destroy()


__all__ = [
    "NativeFolderPicker",
    "NativeFolderPickerBusyError",
    "NativeFolderPickerError",
    "NativeFolderPickerUnavailableError",
    "WindowsNativeFolderPicker",
]
