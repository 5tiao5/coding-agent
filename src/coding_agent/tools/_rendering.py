"""Small rendering primitives shared by model-facing local tools."""

from __future__ import annotations

from unicodedata import category

_DISPLAY_CONTROL_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
_RENDERED_ESCAPE_LENGTHS = {"x": 4, "u": 6, "U": 10}
_LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")
_SUMMARY_PATH_CHARS = 180


def is_display_control(character: str) -> bool:
    """Return whether a character must be escaped before model-visible display."""
    return category(character) in _DISPLAY_CONTROL_CATEGORIES


def render_visible_text(text: str) -> str:
    """Escape control and formatting characters without changing printable text."""
    return "".join(_escape_character(character) for character in text)


def summarize_path(path: str) -> str:
    """Keep a bounded, suffix-preserving path for model-visible summaries."""
    if len(path) <= _SUMMARY_PATH_CHARS:
        return path
    return "..." + path[-(_SUMMARY_PATH_CHARS - 3) :]


def clip_with_ellipsis(text: str, max_chars: int) -> str:
    """Clip already-rendered text to a hard character budget."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def clip_at_escape_boundary(text: str, max_chars: int) -> str:
    """Clip text without splitting one of our rendered control escape tokens."""
    index = 0
    while index < len(text):
        token_length = 1
        if text[index] == "\\" and index + 1 < len(text):
            candidate_length = _RENDERED_ESCAPE_LENGTHS.get(text[index + 1])
            if candidate_length is not None:
                candidate = text[index + 2 : index + candidate_length]
                if len(candidate) == candidate_length - 2 and all(
                    character in _LOWERCASE_HEX_DIGITS for character in candidate
                ):
                    token_length = candidate_length
        if index + token_length > max_chars:
            break
        index += token_length
    return text[:index]


def _escape_character(character: str) -> str:
    if not is_display_control(character):
        return character
    codepoint = ord(character)
    if codepoint <= 0xFF:
        return f"\\x{codepoint:02x}"
    if codepoint <= 0xFFFF:
        return f"\\u{codepoint:04x}"
    return f"\\U{codepoint:08x}"
