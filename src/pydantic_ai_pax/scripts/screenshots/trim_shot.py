"""Trim the empty rows below Pax's status row from a tall screenshot, keeping the window frame."""

import sys

from PIL import Image

FRAME = 60  # The bottom edge of the VHS window, kept as is.
PAD = 70  # Background kept between the status row and the frame.


def trim(path: str) -> None:
    """Cut the empty background between the last row with content and the window frame down to `PAD`."""
    image = Image.open(path).convert('RGB')
    width, height = image.size
    data = image.tobytes()

    def pixel(x: int, y: int) -> bytes:
        start = (y * width + x) * 3
        return data[start : start + 3]

    background = pixel(width // 2, height - FRAME - 10)

    def has_content(y: int) -> bool:
        return any(sum(abs(a - b) for a, b in zip(pixel(x, y), background)) > 30 for x in range(60, width - 60, 2))

    last = max(y for y in range(height - FRAME) if has_content(y))
    trimmed = Image.new('RGB', (width, last + PAD + FRAME))
    trimmed.paste(image.crop((0, 0, width, last + PAD)), (0, 0))
    trimmed.paste(image.crop((0, height - FRAME, width, height)), (0, last + PAD))
    trimmed.save(path)


if __name__ == '__main__':
    trim(sys.argv[1])
