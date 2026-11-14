"""Local image attachments for the prompt editor, not a plugin extension point."""

import os
import platform
import re
import shlex
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import uuid4

import anyio
from PIL import Image, ImageGrab
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

from pydantic_ai.messages import BinaryContent
from pydantic_clai2.runtime import image_codec
from pydantic_clai2.runtime.image_codec import FORMATS, encode_image

MARKER = re.compile(r'\[image:[0-9a-f]{8}\]')
ATTACHED = 'Image attached. Enter sends it; delete its marker to remove it.'


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
    if path.stat().st_size > image_codec.MAX_IMAGE_BYTES:
        raise ValueError('Image file exceeds the 10 MiB attachment limit.')
    with Image.open(path, formats=FORMATS) as image:
        return encode_image(image)


def read_images(paths: Sequence[Path]) -> list[BinaryContent]:
    """Bound a batch while reading, before all of its images can accumulate in memory."""
    images: list[BinaryContent] = []
    size = 0
    for path in paths:
        image = read_image(path)
        size += len(image.data)
        if size > image_codec.MAX_PENDING_BYTES:
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
    try:
        content = ImageGrab.grabclipboard()
    except (NotImplementedError, ChildProcessError) as exc:
        if sys.platform != 'linux':
            raise
        guidance = clipboard_guidance(exc)
        raise type(exc)(f'{exc}. {guidance}') from exc
    if isinstance(content, Image.Image):
        with content:
            return [encode_image(content)]
    if isinstance(content, list) and content:
        return read_images([Path(path) for path in content])
    raise ValueError('No image in the clipboard. Copy an image or paste an image file path.')


def clipboard_guidance(exc: NotImplementedError | ChildProcessError) -> str:
    """Explain a failed Linux backend without probing or replacing its transport."""
    remote = (
        'Run in an accessible graphical session, check your remote clipboard bridge if using SSH, '
        'or paste an image file path accessible on this machine.'
    )
    if isinstance(exc, ChildProcessError):
        return remote
    if os.environ.get('WAYLAND_DISPLAY'):
        package = 'wl-clipboard'
    elif os.environ.get('DISPLAY'):
        package = 'xclip'
    else:
        return remote
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        release = {}
    distributions = {release.get('ID', ''), *release.get('ID_LIKE', '').split()}
    installers = (
        ({'debian', 'ubuntu'}, 'apt-get', 'sudo apt-get install'),
        ({'fedora', 'rhel', 'centos'}, 'dnf', 'sudo dnf install'),
        ({'arch'}, 'pacman', 'sudo pacman -S'),
        ({'alpine'}, 'apk', 'sudo apk add'),
    )
    for family, manager, command in installers:
        if distributions & family and shutil.which(manager):
            return f'Install `{package}` with `{command} {package}`. {remote}'
    return f"Install `{package}` using your distribution's package manager. {remote}"


class ImageInput:
    """Own pending bytes until their visible markers are submitted or discarded."""

    def __init__(self) -> None:
        """Start with no clipboard access and no attachments."""
        self.pending: dict[str, BinaryContent] = {}
        self.notice = ''
        self._error: tuple[str, float] | None = None
        self._error_clock: Callable[[], float] = time.monotonic
        self.retry_text = ''

    def paste_failed(self, exc: Exception, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Give each paste error its own thirty-second lifetime."""
        self.notice = f'Image paste failed: {exc}'
        self._error_clock = clock
        self._error = (self.notice, clock() + 30)

    def expire_notice(self) -> bool:
        """Return whether an expired paste error was cleared, so a live editor can repaint."""
        if self._error is None:
            return False
        message, deadline = self._error
        if self.notice != message:
            self._error = None
        elif self._error_clock() >= deadline:
            self.notice = ''
            self._error = None
            return True
        return False

    def notice_remaining(self) -> float:
        """Bound the existing repaint wait by the paste error's deadline."""
        if self._error is None or self.notice != self._error[0]:
            return float('inf')
        return max(0, self._error[1] - self._error_clock())

    def resume_notice(self, app: Application[str]) -> None:
        """Schedule remaining paste-error expiry on `app`, on failure and each line-prompt reset."""
        self.expire_notice()
        if self._error is None:
            return
        remaining = self.notice_remaining()

        async def expire_feedback() -> None:
            await anyio.sleep(remaining)
            self.expire_notice()
            app.invalidate()

        app.create_background_task(expire_feedback())

    def attach(self, images: Sequence[BinaryContent]) -> str:
        """Stage all images atomically; the editor inserts these removable markers."""
        size = sum(len(image.data) for image in (*self.pending.values(), *images))
        if size > image_codec.MAX_PENDING_BYTES:
            raise ValueError('Pending images exceed 32 MiB. Submit or clear the existing attachments first.')
        markers: list[str] = []
        for image in images:
            marker = f'[image:{uuid4().hex[:8]}]'
            while marker in self.pending:
                marker = f'[image:{uuid4().hex[:8]}]'
            self.pending[marker] = image
            markers.append(marker)
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
        """Ctrl+V reads image data; bracketed text paste only recognizes whole paths."""
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
                self.notice = ATTACHED
            except (OSError, ValueError, NotImplementedError, Image.DecompressionBombError) as exc:
                self.paste_failed(exc)
                self.resume_notice(event.app)
            finally:
                event.app.invalidate()

        return keys
