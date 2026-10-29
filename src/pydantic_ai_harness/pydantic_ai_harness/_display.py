"""Helpers shared by the harness's displays and `ComputerUse`."""

from __future__ import annotations as _annotations

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def png_size(png: bytes) -> tuple[int, int] | None:
    """Width and height from a PNG's header chunk, which the signature puts at bytes 16 to 24."""
    if len(png) < 24 or not png.startswith(PNG_SIGNATURE) or png[12:16] != b'IHDR':
        return None
    return int.from_bytes(png[16:20], 'big'), int.from_bytes(png[20:24], 'big')
