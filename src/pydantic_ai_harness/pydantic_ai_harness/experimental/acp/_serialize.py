"""Coerce tool input/output into JSON-able payloads sized for ACP `session/update` notifications."""

from __future__ import annotations

import json
from collections.abc import Iterator

from pydantic_core import to_jsonable_python

# ACP stdio is newline-delimited JSON, and clients read it with a bounded buffer (asyncio's
# default `StreamReader` limit is 64 KiB). A single oversized notification overruns that buffer
# and drops the connection, so long text is split across several updates.
#
# The SDK serializes outbound text with `json.dumps(..., ensure_ascii=True)`, so a non-ASCII code
# point expands *inside* the JSON string: a BMP char to `\uXXXX` (6 bytes) and an astral char to a
# surrogate pair `\uXXXX\uXXXX` (12 bytes). A character-count cap therefore can't bound the wire
# size -- 8K emoji serialize to ~96 KiB and drop the connection. We chunk by escaped byte length
# instead (see `chunk_text`), leaving headroom below 64 KiB for the notification envelope.
MAX_TEXT_UPDATE_BYTES = 48 * 1024

# Tool input/output is sent whole in one notification (it cannot be chunked across updates like
# streamed text), so an oversized payload is truncated to keep the notification under the buffer.
MAX_RAW_FIELD_CHARS = 16 * 1024


def _escaped_len(char: str) -> int:
    """Bytes `char` occupies inside a `json.dumps(..., ensure_ascii=True)` string."""
    if char in '"\\\b\f\n\r\t':
        return 2  # short escapes: `\"`, `\\`, `\n`, ...
    code = ord(char)
    if code < 0x20:
        return 6  # other control chars -> `\u00XX`
    if code < 0x80:
        return 1  # printable ASCII
    if code < 0x10000:
        return 6  # BMP non-ASCII -> `\uXXXX`
    return 12  # astral plane -> surrogate pair `\uXXXX\uXXXX`


def chunk_text(text: str, budget: int = MAX_TEXT_UPDATE_BYTES) -> Iterator[str]:
    """Split `text` so each chunk's JSON-escaped byte length stays within `budget`.

    Bounds the serialized size of each `session/update` regardless of how the text escapes, so a
    single notification can't overrun the client's read buffer (see `MAX_TEXT_UPDATE_BYTES`).
    """
    chunk: list[str] = []
    size = 0
    for char in text:
        char_size = _escaped_len(char)
        if chunk and size + char_size > budget:
            yield ''.join(chunk)
            chunk = []
            size = 0
        chunk.append(char)
        size += char_size
    if chunk:
        yield ''.join(chunk)


def jsonable(value: object) -> object:
    """Coerce arbitrary tool input/output into something an ACP `session/update` can serialize."""
    # `bytes_mode='base64'` keeps raw (non-UTF-8) bytes from raising; `fallback=str` covers types
    # pydantic cannot otherwise serialize.
    return to_jsonable_python(value, fallback=str, bytes_mode='base64')


def bounded_jsonable(value: object) -> object:
    """`jsonable`, but replace an oversized payload with a truncated marker (see `MAX_RAW_FIELD_CHARS`)."""
    payload = jsonable(value)
    serialized = json.dumps(payload)
    if len(serialized) <= MAX_RAW_FIELD_CHARS:
        return payload
    return {'truncated': True, 'original_length': len(serialized), 'preview': serialized[:MAX_RAW_FIELD_CHARS]}
