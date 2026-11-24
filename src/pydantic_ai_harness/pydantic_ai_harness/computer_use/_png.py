"""PNG helpers for the screenshots a `computer` tool returns."""

from __future__ import annotations as _annotations

from collections.abc import Sequence

from pydantic_ai import messages as _messages
from pydantic_ai.computer_use import ComputerReturnPart

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def png_size(data: bytes) -> tuple[int, int] | None:
    """Width and height from a PNG's header chunk, which the signature puts at bytes 16 to 24."""
    if len(data) < 24 or not data.startswith(PNG_SIGNATURE) or data[12:16] != b'IHDR':
        return None
    return int.from_bytes(data[16:20], 'big'), int.from_bytes(data[20:24], 'big')


def latest_screenshot_size(messages: Sequence[_messages.ModelMessage]) -> tuple[int, int] | None:
    """The size of the latest screenshot a `computer` call returned in `messages`."""
    for message in reversed(messages):
        if not isinstance(message, _messages.ModelRequest):
            continue
        for part in reversed(message.parts):
            if isinstance(part, ComputerReturnPart) and (screenshot := part.screenshot) is not None:
                return png_size(screenshot.data)
    return None
