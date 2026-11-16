"""Formatting, hashing, and editing file content for the filesystem tools."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass

from pydantic_ai.exceptions import ModelRetry


@dataclass
class Replacement:
    """One exact replacement: `old_text` must occur exactly once and is replaced by `new_text`."""

    _: KW_ONLY
    old_text: str
    new_text: str


NOTICE_CHARS = 200
"""Room reserved for the longest continuation or oversized-line notice `format_lines` appends."""


def format_lines(lines: Sequence[str], offset: int, limit: int, max_chars: int | None = None) -> str:
    """Format pre-split lines with line numbers and continuation hint.

    With `max_chars`, the numbered lines end on the last complete one that
    fits, so the continuation offset always names the first line not shown.
    The notice appended after them is not counted; callers reserve `NOTICE_CHARS`.
    """
    total = len(lines)

    if total == 0:
        return '(empty file)\n'

    if offset >= total:
        raise ValueError(f'Offset {offset} exceeds file length ({total} lines).')

    numbered: list[str] = []
    budget = max_chars
    for number, line in enumerate(lines[offset : offset + limit], start=offset + 1):
        rendered = f'{number:>6}\t{line}'
        if budget is not None and len(rendered) > budget:
            if not numbered:
                return (
                    f'... (Line {number} is {len(line):,} characters and does not fit the read window. '
                    f'Use offset={number} to skip it, or a shell byte range to inspect it.)\n'
                )
            break
        numbered.append(rendered)
        if budget is not None:
            budget -= len(rendered)
    result = ''.join(numbered)
    if not result.endswith('\n'):
        result += '\n'

    remaining = total - (offset + len(numbered))
    if remaining > 0:
        next_offset = offset + len(numbered)
        result += f'... ({remaining} more lines. Use offset={next_offset} to continue reading.)\n'

    return result


def is_binary(data: bytes, sample_size: int = 8192) -> bool:
    """Detect binary content by checking for null bytes in the sample."""
    return b'\x00' in data[:sample_size]


MAX_MEDIA_BYTES = 5_000_000
"""Largest file `read_file` returns as file content.

File content bypasses the textual `max_read_chars` budget, and 5 MB is the
strictest per-image limit among mainstream providers. A larger file gets the
text description instead of content that would fail the next model request.
"""

_HEX_PREVIEW_BYTES = 64
"""Leading bytes of a non-media binary file shown as hex."""


def sniff_media_type(data: bytes) -> str | None:
    """The media type of a file `read_file` returns as itself, read from its leading bytes.

    Images, PDFs and common video formats are returned as files. A model that can't
    take one gets a note in its place, or a description from `FileUnderstanding`,
    rather than a failed request. Other formats stay described, because some models
    whose profile admits the kind reject them: Gemini takes no Office documents or
    Matroska, and OpenAI-compatible endpoints report audio input for models that
    reject it. The bytes decide, not the extension, so a misnamed file is not sent
    under the wrong media type.

    The image signatures duplicate core's `image_media_type_from_bytes`
    (`pydantic_ai/images/_media_type.py`), which is private and covers neither GIF
    nor PDF. Change both together until core offers a public helper to use here.
    """
    return _image_or_pdf(data) or _video(data)


def _image_or_pdf(data: bytes) -> str | None:
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif'
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'image/webp'
    if data.startswith(b'%PDF-'):
        return 'application/pdf'
    return None


_MP4_VIDEO_BRANDS = frozenset(
    {b'isom', b'iso2', b'iso3', b'iso4', b'iso5', b'iso6', b'mp41', b'mp42', b'avc1', b'M4V ', b'M4VH', b'M4VP'}
    | {b'dash', b'mmp4', b'MSNV', b'f4v ', b'NDAS'}
)
"""`ftyp` brands of MP4 video."""


def _video(data: bytes) -> str | None:
    if data[4:8] == b'ftyp':
        brand = data[8:12]
        if brand == b'qt  ':
            return 'video/quicktime'
        if brand.startswith((b'3gp', b'3g2')):
            return 'video/3gpp'
        # Only known video brands: HEIF and AVIF photos, JPEG 2000, camera raw files and MPEG-4 audio share the
        # `ftyp` box.
        return 'video/mp4' if brand in _MP4_VIDEO_BRANDS else None
    if data.startswith(b'\x1a\x45\xdf\xa3') and b'webm' in data[:64]:
        # WebM's EBML header names it in its DocType, near the start; Matroska, which shares the header, isn't sent.
        return 'video/webm'
    if data.startswith(b'FLV\x01'):
        return 'video/x-flv'
    if data.startswith((b'\x00\x00\x01\xba', b'\x00\x00\x01\xb3')):
        return 'video/mpeg'
    if data.startswith(b'\x30\x26\xb2\x75\x8e\x66\xcf\x11\xa6\xd9\x00\xaa\x00\x62\xce\x6c'):
        return 'video/x-ms-wmv'
    return None


_HEX_ROW_CHARS = 58
"""Characters in one full hex row: an 8-digit offset, two spaces, 16 bytes, and the newline."""


def hex_preview(data: bytes, max_chars: int | None = None) -> str:
    """Up to the first `_HEX_PREVIEW_BYTES` of `data` as offset-prefixed hex rows of 16 bytes.

    With `max_chars`, only as many rows as fit are shown; the notice for the rest is not counted,
    and callers reserve `NOTICE_CHARS` for it.
    """
    limit = _HEX_PREVIEW_BYTES if max_chars is None else min(_HEX_PREVIEW_BYTES, max_chars // _HEX_ROW_CHARS * 16)
    head = data[:limit]
    rows = [f'{start:08x}  {head[start : start + 16].hex(" ")}\n' for start in range(0, len(head), 16)]
    more = f'... ({len(data) - len(head)} more bytes)\n' if len(data) > len(head) else ''
    return ''.join(rows) + more


def bytes_hash(data: bytes) -> str:
    """The content hash every tool reports: SHA-256 of the file's raw bytes, first 12 hex characters.

    `read_file`, `write_file`, `edit_file`, and `file_info` all hash the bytes
    in the workspace, so a hash from any of them identifies the same content and
    the `expected_hash` handshake holds whatever the line endings or encoding.
    """
    return hashlib.sha256(data).hexdigest()[:12]


def text_hash(content: str) -> str:
    """The hash of `content` as the tools write it: its UTF-8 bytes."""
    return bytes_hash(content.encode('utf-8'))


def check_expected_hash(path: str, current_hash: str, expected_hash: str) -> None:
    """Reject a write or edit whose `expected_hash` no longer matches the file."""
    if current_hash != expected_hash:
        raise ValueError(
            f'Conflict: file {path!r} has changed (expected hash:{expected_hash}, '
            f'got hash:{current_hash}). Re-read the file and retry.'
        )


def normalize_replacements(
    old_text: str | None, new_text: str | None, replacements: Sequence[Replacement] | None
) -> list[Replacement]:
    """Normalize the two argument forms of `edit_file` into one ordered list."""
    if replacements is None:
        if old_text is None or new_text is None:
            raise ModelRetry('Provide old_text and new_text, or a non-empty replacements list.')
        return [Replacement(old_text=old_text, new_text=new_text)]
    if old_text is not None or new_text is not None or not replacements:
        raise ModelRetry('Provide either old_text and new_text or a non-empty replacements list, not both.')
    return list(replacements)


def apply_replacements(text: str, replacements: Sequence[Replacement], path: str) -> str:
    """Apply `replacements` in order, each matching exactly once; nothing is written on failure."""
    for index, replacement in enumerate(replacements, start=1):
        label = f'replacement {index}' if len(replacements) > 1 else 'old_text'
        if not replacement.old_text:
            raise ValueError(f'{label} is empty; old_text must be the exact text to replace.')
        count = text.count(replacement.old_text)
        if count == 0:
            raise ValueError(f'{label} not found in {path}. No changes were written.')
        if count > 1:
            raise ValueError(
                f'{label} found {count} times in {path}. Include more surrounding context to make the match '
                'unique. No changes were written.'
            )
        text = text.replace(replacement.old_text, replacement.new_text, 1)
    return text
