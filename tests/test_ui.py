"""Tests for terminal-safe event rendering helpers."""

from __future__ import annotations

from io import BytesIO, TextIOWrapper

from rich.console import Console

from coding_agent.ui import console_safe


def test_console_safe_replaces_characters_unsupported_by_the_console_encoding() -> None:
    ascii_stream = TextIOWrapper(BytesIO(), encoding="ascii")
    console = Console(file=ascii_stream, force_terminal=False, color_system=None)

    assert console_safe("plain Ω text", console) == "plain ? text"
