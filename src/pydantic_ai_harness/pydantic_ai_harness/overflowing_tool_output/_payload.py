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
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _sketch_mapping(mapping: Mapping[object, object]) -> str:
    keys = list(mapping)
    shown = ', '.join(f'{key!r}: {type(mapping[key]).__name__}' for key in keys[:10])
    more = '' if len(keys) <= 10 else f', ... ({len(keys)} keys)'
    return f'{{{shown}{more}}}'


def _sketch_sequence(items: Sequence[object]) -> str:
    elem = type(items[0]).__name__ if items else 'empty'
    return f'[{len(items)} items of {elem}]'


def truncate_text(text: str, max_chars: int, strategy: TruncationStrategy) -> str:
    """Cut `text` down to roughly `max_chars`, annotating what was removed.

    Returns `text` unchanged when it already fits.
    """
    total = len(text)
    if total <= max_chars:
        return text

    if strategy is TruncationStrategy.head:
        return f'{text[:max_chars]}\n\n[truncated: showing first {max_chars:,} of {total:,} chars]'
    if strategy is TruncationStrategy.tail:
        return f'[truncated: showing last {max_chars:,} of {total:,} chars]\n\n{text[-max_chars:]}'

    head_chars = max_chars * 2 // 5
    tail_chars = max_chars - head_chars
    omitted = total - head_chars - tail_chars
    return (
        f'{text[:head_chars]}\n\n'
        f'[truncated: {omitted:,} chars omitted from the middle; '
        f'showing first {head_chars:,} + last {tail_chars:,} of {total:,} chars]\n\n'
        f'{text[-tail_chars:]}'
    )
