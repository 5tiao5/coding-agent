"""Byte-preserving text codec tests."""

from __future__ import annotations

import pytest

from coding_agent.text import (
    TextDocumentError,
    decode_utf8_document,
    detect_newline_style,
    encode_utf8_document,
    normalize_argument_text,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain", "none"),
        ("a\nb\n", "lf"),
        ("a\r\nb\r\n", "crlf"),
        ("a\r\nb\n", "mixed"),
        ("a\rb", "mixed"),
    ],
)
def test_detect_newline_style(text: str, expected: str) -> None:
    assert detect_newline_style(text) == expected


def test_document_decode_preserves_byte_facts_and_normalizes_crlf() -> None:
    raw = b"\xef\xbb\xbffirst\r\nsecond"

    document = decode_utf8_document(raw)

    assert document.raw == raw
    assert document.text == "first\nsecond"
    assert document.newline == "crlf"
    assert document.utf8_bom is True
    assert document.ends_with_newline is False
    assert len(document.sha256) == 64
    assert encode_utf8_document(document.text, newline="crlf", utf8_bom=document.utf8_bom) == raw


@pytest.mark.parametrize(
    ("raw", "reason"),
    [(b"a\x00b", "binary"), (b"\xff", "unsupported_encoding")],
)
def test_document_decode_rejects_non_text(raw: bytes, reason: str) -> None:
    with pytest.raises(TextDocumentError) as caught:
        decode_utf8_document(raw)

    assert caught.value.reason == reason


def test_argument_newlines_are_normalized_but_bare_carriage_return_is_rejected() -> None:
    assert normalize_argument_text("a\r\nb") == "a\nb"

    with pytest.raises(ValueError, match="bare carriage"):
        normalize_argument_text("a\rb")
