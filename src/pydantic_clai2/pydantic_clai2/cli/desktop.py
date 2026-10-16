"""`clai2 --desktop`: the web UI in a window of its own instead of a browser tab.

The window needs its process's main thread, which CLAI's event loop holds, so `open_window` runs this
module as a child process. The child reads the page's address, token included, from standard input, so
the token never appears on a command line, shows it with pywebview, and exits when the window closes.
"""

import subprocess
import sys
import webbrowser
from collections.abc import Sequence
from importlib.resources import files
from importlib.util import find_spec
from typing import TextIO
from urllib.parse import urlsplit

import anyio

from pydantic_clai2.config.settings_store import state_dir

TITLE = 'CLAI'
MISSING = (
    "--desktop needs pywebview: install CLAI with its desktop extra, such as uv tool install 'pydantic-clai2[desktop]'"
)


def require_webview() -> None:
    """Raise `ValueError` naming the extra unless pywebview is installed."""
    if find_spec('webview') is None:
        raise ValueError(MISSING)


async def open_window(url: str, *, command: Sequence[str] | None = None) -> int:
    """Show `url` in a desktop window, returning the window process's exit status once the user closes it.

    `command` replaces the child process, `python -m pydantic_clai2.cli.desktop`. Cancelling kills the window.
    """
    argv = [sys.executable, '-P', '-m', __name__] if command is None else list(command)
    process = await anyio.open_process(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=None)
    async with process:
        assert process.stdin is not None
        await process.stdin.send(f'{url}\n'.encode())
        await process.stdin.aclose()
        return await process.wait()


class Bridge:
    """What the page may ask of its window, as `window.pywebview.api.open_external(url)`."""

    def open_external(self, url: str) -> bool:
        """Open an `http` or `https` address in the default browser, as a link with `target="_blank"` does."""
        if urlsplit(url).scheme not in ('http', 'https'):
            return False
        return webbrowser.open(url)


def main(stdin: TextIO = sys.stdin) -> None:
    """Show the web UI address on `stdin` in a window until the user closes it."""
    url = stdin.readline().strip()
    if urlsplit(url).scheme != 'http':
        raise SystemExit('Expected the web UI address on standard input.')
    import webview

    storage = state_dir() / 'desktop'
    storage.mkdir(parents=True, exist_ok=True)
    # pywebview turns text selection off by default, which would also stop copying from the transcript.
    webview.create_window(  # pyright: ignore[reportUnknownMemberType]
        TITLE, url, js_api=Bridge(), maximized=True, min_size=(800, 500), text_select=True
    )
    # Not private: the page keeps its preferences (theme, sidebar, panels) in local storage across launches.
    icon = files('pydantic_clai2.builtin_plugins.web') / 'static' / 'clai-icon.png'
    webview.start(private_mode=False, storage_path=str(storage), icon=str(icon))


if __name__ == '__main__':
    main()
