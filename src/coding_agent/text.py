"""Small, provider-neutral helpers for safe UTF-8 text mutations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

NewlineStyle = Literal["none", "lf", "crlf", "mixed"]
WritableNewlineStyle = Literal["lf", "crlf"]

_UTF8_BOM = b"\xef\xbb\xbf"


class TextDocumentError(ValueError):
    """A stable classification for bytes that are unsafe to treat as editable text."""

    def __init__(self, reason: Literal["binary", "unsupported_encoding"]) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class TextDocument:
    """Decoded text plus the byte-level facts needed for lossless rewrites."""

    raw: bytes
    text: str
    sha256: str
    newline: NewlineStyle
    utf8_bom: bool
    ends_with_newline: bool


def decode_utf8_document(data: bytes) -> TextDocument:
    """Decode UTF-8 bytes and normalize a uniform CRLF document to logical LF."""
    if b"\x00" in data:
        raise TextDocumentError("binary")

    has_bom = data.startswith(_UTF8_BOM)
    payload = data[len(_UTF8_BOM) :] if has_bom else data
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TextDocumentError("unsupported_encoding") from exc

    newline = detect_newline_style(decoded)
    logical_text = decoded.replace("\r\n", "\n") if newline == "crlf" else decoded
    return TextDocument(
        raw=data,
        text=logical_text,
        sha256=sha256(data).hexdigest(),
        newline=newline,
        utf8_bom=has_bom,
        ends_with_newline=decoded.endswith(("\n", "\r")),
    )


def detect_newline_style(text: str) -> NewlineStyle:
    """Classify line endings without silently accepting bare carriage returns."""
    crlf_count = text.count("\r\n")
    without_crlf = text.replace("\r\n", "")
    lf_count = without_crlf.count("\n")
    bare_cr_count = without_crlf.count("\r")
    styles = sum(count > 0 for count in (crlf_count, lf_count, bare_cr_count))
    if styles == 0:
        return "none"
    if styles > 1 or bare_cr_count:
        return "mixed"
    return "crlf" if crlf_count else "lf"


def normalize_argument_text(text: str) -> str:
    """Convert JSON argument CRLF to logical LF and reject ambiguous bare CR."""
    normalized = text.replace("\r\n", "\n")
    if "\r" in normalized:
        raise ValueError("text contains unsupported bare carriage returns")
    return normalized


def encode_utf8_document(
    logical_text: str,
    *,
    newline: WritableNewlineStyle,
    utf8_bom: bool,
) -> bytes:
    """Encode logical LF text using an explicit on-disk newline convention."""
    rendered = logical_text.replace("\n", "\r\n") if newline == "crlf" else logical_text
    payload = rendered.encode("utf-8")
    return (_UTF8_BOM + payload) if utf8_bom else payload
