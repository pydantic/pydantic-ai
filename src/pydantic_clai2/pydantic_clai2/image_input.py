"""Local image attachments for the prompt editor, not a plugin extension point."""

import re
import shlex
from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageGrab, ImageOps
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from pydantic_ai.messages import BinaryContent
from typing_extensions import Buffer

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PENDING_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 25_000_000
MARKER = re.compile(r'\[image:[0-9a-f]{8}\]')


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


def local_image_path(text: str) -> Path:
    """Reject network and device namespaces before probing the filesystem."""
    if text.replace('\\', '/').startswith('//'):
        raise ValueError('UNC and device paths are not supported for image attachments.')
    path = Path(text).expanduser()
    if str(path).replace('\\', '/').startswith('//'):
        raise ValueError('UNC and device paths are not supported for image attachments.')
    return path


def read_image(path: Path) -> BinaryContent:
    """Read a local file, checking its contents rather than trusting its suffix."""
    path = local_image_path(str(path))
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError('Image file exceeds the 10 MiB attachment limit.')
    with Image.open(path, formats=['PNG', 'JPEG', 'GIF', 'WEBP', 'BMP', 'TIFF']) as image:
        return encode_image(image)


def read_images(paths: Sequence[Path]) -> list[BinaryContent]:
    """Bound a batch while reading, before all of its images can accumulate in memory."""
    images: list[BinaryContent] = []
    size = 0
    for path in paths:
        image = read_image(path)
        size += len(image.data)
        if size > MAX_PENDING_BYTES:
            raise ValueError('Pasted images exceed the 32 MiB pending attachment limit.')
        images.append(image)
    return images


def pasted_paths(text: str) -> list[Path]:
    """Recognize a paste consisting solely of image paths, including shell quoting."""
    text = text.strip()
    if len(text) > 32768 or '\x00' in text:
        return []
    try:
        # Try the entire path first, so Windows backslashes and unquoted spaces survive.
        path = local_image_path(text.strip('"\''))
        if path.is_file():
            paths = [path]
        else:
            tokens = shlex.split(text, posix='\\' not in text or '\\ ' in text)
            paths = [local_image_path(token.strip('"\'')) for token in tokens]
        suffixes = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif', '.tiff'}
        return paths if paths and all(p.suffix.lower() in suffixes and p.is_file() for p in paths) else []
    except (OSError, ValueError, RuntimeError):
        return []


def clipboard_images() -> list[BinaryContent]:
    """Use Pillow's native macOS/Windows and wl-paste/xclip Linux backends."""
    content = ImageGrab.grabclipboard()
    if isinstance(content, Image.Image):
        with content:
            return [encode_image(content)]
    if isinstance(content, list) and content:
        return read_images([Path(path) for path in content])
    raise ValueError('No image in the clipboard. Copy an image or paste an image file path.')


class ImageInput:
    """Own pending bytes until their visible markers are submitted or discarded."""

    def __init__(self) -> None:
        """Start with no clipboard access and no attachments."""
        self.pending: dict[str, BinaryContent] = {}
        self.notice = ''
        self.retry_text = ''

    def attach(self, images: Sequence[BinaryContent]) -> str:
        """Stage all images atomically; the editor inserts these removable markers."""
        size = sum(len(image.data) for image in (*self.pending.values(), *images))
        if size > MAX_PENDING_BYTES:
            raise ValueError('Pending images exceed 32 MiB. Submit or clear the existing attachments first.')
        markers: list[str] = []
        for image in images:
            marker = f'[image:{uuid4().hex[:8]}]'
            while marker in self.pending:
                marker = f'[image:{uuid4().hex[:8]}]'
            self.pending[marker] = image
            markers.append(marker)
        self.notice = 'Image attached. Enter sends it; delete its marker to remove it.'
        return ' '.join(markers) + ' '

    def resolve(self, text: str) -> tuple[str, list[BinaryContent]]:
        """Snapshot attachments before hooks run; stale history markers fail visibly."""
        markers = MARKER.findall(text)
        if any(marker not in self.pending for marker in markers):
            raise ValueError('This image attachment has expired. Paste the image again.')
        self.retry_text = ''
        return MARKER.sub('', text).strip(), [self.pending[marker] for marker in markers]

    def retain(self, texts: Sequence[str]) -> None:
        """Release bytes not referenced by the draft or queued prompts."""
        markers = {marker for text in (*texts, self.retry_text) for marker in MARKER.findall(text)}
        self.pending = {marker: image for marker, image in self.pending.items() if marker in markers}

    def bindings(self, *, queued: Callable[[], Sequence[str]] = tuple) -> KeyBindings:
        """Ctrl-V reads image data; bracketed text paste only recognizes whole paths."""
        keys = KeyBindings()

        @keys.add('c-v')
        @keys.add('escape', 'v')
        def clipboard(event: KeyPressEvent) -> None:
            insert(event=event, text=None)

        @keys.add('<bracketed-paste>')
        def paste(event: KeyPressEvent) -> None:
            insert(event=event, text=event.data)

        def insert(*, event: KeyPressEvent, text: str | None) -> None:
            buffer = event.current_buffer
            self.retain([buffer.text, *queued()])
            self.notice = ''
            try:
                if text is None:
                    images = clipboard_images()
                else:
                    paths = pasted_paths(text)
                    if not paths:
                        buffer.insert_text(text.replace('\r\n', '\n').replace('\r', '\n'))
                        return
                    images = read_images(paths)
                buffer.insert_text(self.attach(images))
            except (OSError, ValueError, NotImplementedError, Image.DecompressionBombError) as exc:
                self.notice = f'Image paste failed: {exc}. Linux requires wl-paste (Wayland) or xclip (X11).'
            finally:
                event.app.invalidate()

        return keys
