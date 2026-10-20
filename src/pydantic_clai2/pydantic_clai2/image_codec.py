"""Checking and normalizing image bytes for a prompt, wherever they come from: the terminal or the web UI."""

from collections.abc import Sequence
from io import BytesIO

from PIL import Image, ImageOps
from typing_extensions import Buffer

from pydantic_ai.messages import BinaryContent

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PENDING_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 25_000_000
FORMATS = ('PNG', 'JPEG', 'GIF', 'WEBP', 'BMP', 'TIFF')


class ImageBuffer(BytesIO):
    """Reject oversized encoded output before it grows the attachment buffer."""

    def write(self, data: Buffer, /) -> int:
        """Enforce the byte limit for each encoder write, including partial output."""
        if self.tell() + memoryview(data).nbytes > MAX_IMAGE_BYTES:
            raise ValueError('Image exceeds the 10 MiB attachment limit.')
        return super().write(data)


def encode_image(image: Image.Image) -> BinaryContent:
    """Normalize to PNG, bounding decoded dimensions and encoded payload size."""
    if image.width * image.height > MAX_PIXELS:
        raise ValueError('Image exceeds the 25 megapixel limit.')
    mode = 'RGBA' if 'A' in image.getbands() or 'transparency' in image.info else 'RGB'
    with ImageBuffer() as output, ImageOps.exif_transpose(image) as oriented, oriented.convert(mode) as normalized:
        normalized.info.clear()
        normalized.save(output, format='PNG')
        return BinaryContent(data=output.getvalue(), media_type='image/png')


def decode_images(images: Sequence[bytes]) -> list[BinaryContent]:
    """Read image bytes from elsewhere, such as the browser's clipboard, checked and normalized as a file is.

    The batch is bounded before any image is decoded.
    """
    if any(len(data) > MAX_IMAGE_BYTES for data in images):
        raise ValueError('Image exceeds the 10 MiB attachment limit.')
    if sum(len(data) for data in images) > MAX_PENDING_BYTES:
        raise ValueError('Images exceed the 32 MiB pending attachment limit.')
    return [_decode(data) for data in images]


def _decode(data: bytes) -> BinaryContent:
    try:
        with Image.open(BytesIO(data), formats=FORMATS) as image:
            return encode_image(image)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f'Not an image CLAI can read: {exc}') from exc
