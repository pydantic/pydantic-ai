"""Size measurement, stringification, truncation, and binary detection.

Harvested from PR #185 (`ToolOutputManagement`) and adapted: character-based truncation
strategies, ANSI stripping, and binary detection. Token measurement reuses the compaction
heuristic via `estimate_token_count` so the two capabilities stay aligned.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import TypeGuard

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart
from pydantic_core import to_json

from pydantic_ai_harness._output import truncate_tail
from pydantic_ai_harness.compaction._shared import estimate_token_count


class TruncationStrategy(str, Enum):
    """Which end(s) of an oversized text to keep when truncating."""

    head = 'head'
    """Keep the first characters (good for headers / schemas)."""

    tail = 'tail'
    """Keep the last characters (good for build / test output, where errors land last)."""

    head_tail = 'head_tail'
    """Keep the first and last characters, eliding the middle."""


# CSI sequences, OSC sequences, and simple escapes. Terminal tool output is full of color
# codes that waste tokens and can confuse models.
_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b[^[\]()]')


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from `text`."""
    return _ANSI_ESCAPE_RE.sub('', text)


def is_binary(value: object) -> bool:
    """Return True for raw byte payloads, which must never be stringify-truncated."""
    return isinstance(value, (bytes, bytearray, memoryview))


def to_bytes(value: object) -> bytes:
    """Serialize any tool return value to the bytes that get spilled.

    Strings spill as UTF-8 text; byte payloads spill verbatim; everything else spills as
    JSON so the stored payload stays valid and grep-able.
    """
    if isinstance(value, str):
        return value.encode('utf-8')
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return to_json(value)


def to_text(value: object) -> str:
    """Render a non-binary tool return value as the text used for measuring and truncating.

    Strings pass through; structured values become JSON (truncating JSON is lossy, so prefer
    spill or summarize for them -- see the README).
    """
    if isinstance(value, str):
        return value
    return to_json(value).decode('utf-8', errors='replace')


# A serializer callable for structured (non-string, non-binary) tool returns:
# `(value) -> text`, where the text is what gets measured, previewed, spilled, and read back.
Serializer = Callable[[object], str]

# `str.splitlines` treats these as line breaks, but JSON only requires escaping control
# characters below 0x20, so `to_json` emits them raw inside string values. Escaping them
# keeps read-back line slicing aligned with the lines the presets rendered.
_LINE_SEPARATOR_ESCAPES = {0x85: '\\u0085', 0x2028: '\\u2028', 0x2029: '\\u2029'}


def indented_json(value: object) -> str:
    """Serializer preset: render `value` as indented JSON, one field per line.

    A spilled payload rendered this way can be paged and `pattern`-filtered by line
    through `read_tool_result`; compact JSON puts the whole value on one line.
    """
    return to_json(value, indent=2).decode('utf-8', errors='replace').translate(_LINE_SEPARATOR_ESCAPES)


def json_lines(value: object) -> str:
    """Serializer preset: render a sequence as JSON Lines, one compact JSON value per line.

    Line N is item N, so `read_tool_result` offsets and limits map directly to items and a
    `pattern` match returns whole items; an empty sequence renders as an empty string.
    Values that are not sequences (or that are strings or byte payloads) fall back to
    `indented_json`.
    """
    if _is_text_sequence(value):
        rendered = '\n'.join(to_json(item).decode('utf-8', errors='replace') for item in value)
        return rendered.translate(_LINE_SEPARATOR_ESCAPES)
    return indented_json(value)


def measure(text: str, *, over_tokens: bool, tokenizer: Callable[[str], int] | None) -> int:
    """Measure `text` in characters (default) or estimated tokens (`over_tokens=True`)."""
    if not over_tokens:
        return len(text)
    message: ModelMessage = ModelRequest(parts=[SystemPromptPart(content=text)])
    return estimate_token_count([message], tokenizer)


def json_sketch(value: object) -> str:
    """Build a one-line shape hint for a structured value, or '' for anything else.

    The `_is_*` guards are `TypeGuard`s, so a `Mapping`/`Sequence` value narrows to a known
    element type (`object`) the strict type checker accepts -- no `Any` and no `Unknown`.
    """
    if _is_mapping(value):
        return _sketch_mapping(value)
    if _is_text_sequence(value):
        return _sketch_sequence(value)
    return ''


def _is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_text_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview))


def _sketch_mapping(mapping: Mapping[object, object]) -> str:
    keys = list(mapping)
    shown = ', '.join(f'{key!r}: {type(mapping[key]).__name__}' for key in keys[:10])
    more = '' if len(keys) <= 10 else f', ... ({len(keys)} keys)'
    return f'{{{shown}{more}}}'


def _sketch_sequence(items: Sequence[object]) -> str:
    elem = type(items[0]).__name__ if items else 'empty'
    return f'[{len(items)} items of {elem}]'


def _tail_line_start(text: str, count: int) -> int:
    """Find the last N LF/CRLF lines without splitting the entire payload."""
    end = len(text) - int(text.endswith('\n'))
    for _ in range(count):
        end = text.rfind('\n', 0, end)
        if end < 0:
            return 0
    return end + 1


def truncate_text(text: str, max_chars: int, strategy: TruncationStrategy, *, keep_tail_lines: int = 0) -> str:
    """Limit `text` to `max_chars`, including the truncation marker.

    Omit the marker when it cannot fit alongside the retained content. Reserved tail
    lines take priority over both the marker and the remaining content.
    """
    if max_chars <= 0:
        return ''
    total = len(text)
    if total <= max_chars:
        return text
    tail_chars = total - _tail_line_start(text, keep_tail_lines) if keep_tail_lines else 0
    if tail_chars > max_chars:
        return truncate_tail(text, max_chars)
    if tail_chars == max_chars:
        return text[-max_chars:]
    if strategy is TruncationStrategy.tail:
        truncated = truncate_tail(text, max_chars)
        if tail_chars and not truncated.endswith(text[total - tail_chars :]):
            return text[-max_chars:]
        return truncated

    return _truncate_head_and_tail(text, max_chars, strategy, tail_chars)


def _truncate_head_and_tail(text: str, max_chars: int, strategy: TruncationStrategy, tail_chars: int) -> str:
    total = len(text)
    head_share = 5 if strategy is TruncationStrategy.head else 2

    # Check each count: digit and thousands-separator changes can shorten the marker.
    for retained in range(max_chars, max(1, tail_chars) - 1, -1):
        head_chars = (retained - tail_chars) * head_share // 5
        if strategy is TruncationStrategy.head and not tail_chars:
            marker = f'\n\n[truncated: showing first {retained:,} of {total:,} chars]'
        else:
            marker = (
                f'\n\n[truncated: {total - retained:,} chars omitted from the middle; '
                f'showing first {head_chars:,} + last {retained - head_chars:,} of {total:,} chars]\n\n'
            )
        if retained + len(marker) <= max_chars:
            break
    else:
        if tail_chars:
            return text[-max_chars:]
        retained, marker = max_chars, ''
        head_chars = retained * head_share // 5

    return text[:head_chars] + marker + text[total - (retained - head_chars) :]
