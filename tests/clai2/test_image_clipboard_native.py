"""Opt-in desktop integration: these tests replace the system clipboard."""

import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from pydantic_clai2.ui.prompt.image_input import clipboard_images, read_image
from pydantic_clai2.ui.prompt.text_clipboard import copy_command, run_copy


@pytest.mark.skipif(os.environ.get('CLAI_TEST_CLIPBOARD') != '1', reason='requires an isolated desktop clipboard')
def test_native_text_copy() -> None:
    """A drag-selection's copy reaches the clipboard through the platform's own command."""
    command = copy_command()  # Imported before the suite's fixture replaces it.
    assert command is not None
    text = 'copied from CLAI \u2713\nsecond line'
    run_copy(command=command, text=text)
    if sys.platform == 'darwin':
        paste = ['pbpaste']
    elif sys.platform == 'win32':
        paste = [
            'powershell',
            '-NoProfile',
            '-Command',
            '[Console]::OutputEncoding = [Text.Encoding]::UTF8; Get-Clipboard -Raw',
        ]
    else:
        paste = ['xclip', '-selection', 'clipboard', '-o']
    environment = {**os.environ, 'LC_CTYPE': 'UTF-8'}  # `pbpaste` writes Mac Roman without it.
    pasted = subprocess.run(paste, check=True, timeout=15, capture_output=True, env=environment).stdout.decode()
    assert pasted.replace('\r\n', '\n').rstrip('\n') == text


@pytest.mark.skipif(os.environ.get('CLAI_TEST_CLIPBOARD') != '1', reason='requires an isolated desktop clipboard')
def test_native_clipboard(tmp_path: Path) -> None:
    path = tmp_path / 'clipboard.png'
    Image.new('RGB', (3, 2), color='red').save(path)
    if sys.platform == 'darwin':
        command = ['osascript', '-e', f'set the clipboard to (read POSIX file "{path}" as «class PNGf»)']
    elif sys.platform == 'win32':
        command = [
            'powershell',
            '-NoProfile',
            '-STA',
            '-Command',
            'Add-Type -AssemblyName System.Windows.Forms; '
            'Add-Type -AssemblyName System.Drawing; '
            f"$image = [System.Drawing.Image]::FromFile('{str(path).replace(chr(39), chr(39) * 2)}'); "
            '[System.Windows.Forms.Clipboard]::SetImage($image); $image.Dispose()',
        ]
    else:
        command = ['xclip', '-selection', 'clipboard', '-t', 'image/png', '-i', str(path)]
    # xclip forks a clipboard owner; do not wait for an inherited stderr pipe to close.
    subprocess.run(command, check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    images = clipboard_images()
    assert len(images) == 1
    # Some clipboard backends supply RGBA even when the source is RGB.
    with Image.open(BytesIO(images[0].data)) as received, Image.open(BytesIO(read_image(path).data)) as original:
        assert received.convert('RGB').tobytes() == original.convert('RGB').tobytes()
        assert received.size == original.size
