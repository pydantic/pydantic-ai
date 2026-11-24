"""What every live desktop's display must do, checked once per backend against a real screen."""

from __future__ import annotations as _annotations

import zlib

from pydantic_ai.computer_use import MouseMove, Point, Region

from .display_fakes import FullDisplay


def png_size(png: bytes) -> tuple[int, int]:
    """Width and height from a PNG's header chunk, at bytes 16 to 24."""
    return int.from_bytes(png[16:20], 'big'), int.from_bytes(png[20:24], 'big')


async def check_display_contract(display: FullDisplay) -> None:
    """Check that screenshots match the display size, zoom gives the size asked for, and the pointer moves."""
    width, height = await display.display_size()
    assert png_size(await display.screenshot()) == (width, height)
    assert png_size(await display.zoom(Region(left=10, top=20, right=110, bottom=70))) == (100, 50)
    await display.act(MouseMove(x=width // 3, y=height // 3))
    assert await display.cursor_position() == Point(x=width // 3, y=height // 3)


def is_blank(png: bytes) -> bool:
    """Whether every pixel of a PNG has the same color, as a screen with nothing drawn on it does.

    Decodes the 8-bit, non-interlaced RGB or RGBA PNGs that `scrot` writes, with the standard library only.
    """
    assert png[24] == 8 and png[28] == 0, 'expected 8 bits per channel, not interlaced'
    width, _ = png_size(png)
    step = {2: 3, 6: 4}[png[25]]  # bytes per pixel: RGB or RGBA
    compressed = bytearray()
    offset = 8  # past the signature
    while offset < len(png):
        length = int.from_bytes(png[offset : offset + 4], 'big')
        if png[offset + 4 : offset + 8] == b'IDAT':
            compressed += png[offset + 8 : offset + 8 + length]
        offset += 12 + length  # length, type, data and CRC
    rows = zlib.decompress(compressed)
    stride = width * step
    above = bytearray(stride)
    for start in range(0, len(rows), stride + 1):
        kind, row = rows[start], bytearray(rows[start + 1 : start + 1 + stride])
        for i in range(stride):
            left = row[i - step] if i >= step else 0
            if kind == 1:
                row[i] = (row[i] + left) & 0xFF
            elif kind == 2:
                row[i] = (row[i] + above[i]) & 0xFF
            elif kind == 3:
                row[i] = (row[i] + (left + above[i]) // 2) & 0xFF
            elif kind == 4:  # predict from the neighbor nearest `left + up - up_left`; ties go to left, then up
                up, up_left = above[i], above[i - step] if i >= step else 0
                guess = left + up - up_left
                nearest = min(
                    (abs(guess - left), 0, left), (abs(guess - up), 1, up), (abs(guess - up_left), 2, up_left)
                )
                row[i] = (row[i] + nearest[2]) & 0xFF
        if row != row[:step] * width or (start and row[:step] != above[:step]):
            return False
        above = row
    return True
