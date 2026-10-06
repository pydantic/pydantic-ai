"""Copy text out of CLAI: the local clipboard command, or OSC 52 for the terminal to set it."""

import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field, replace
from typing import IO

from termflow.ansi import make_clipboard_copy

COPY_TIMEOUT = 5.0
"""Seconds a clipboard command may take before the copy is abandoned."""


@dataclass(frozen=True, kw_only=True)
class ClipboardCommand:
    """A local command that sets the system clipboard from its standard input."""

    argv: tuple[str, ...]
    encoding: str = 'utf-8'
    env: dict[str, str] = field(default_factory=dict[str, str])
    """Variables set for the command on top of CLAI's own environment."""


def copy_command(*, platform: str = sys.platform) -> ClipboardCommand | None:
    """The installed clipboard command for this machine, or `None` when there is none."""
    candidates: list[ClipboardCommand] = []
    if platform == 'darwin':
        # Without a UTF-8 locale, `pbcopy` reads its input as Mac Roman.
        candidates.append(ClipboardCommand(argv=('pbcopy',), env={'LC_CTYPE': 'UTF-8'}))
    elif platform == 'win32':
        # The system's own `clip.exe`: a `PATH` search on Windows looks in the working directory
        # first, which may be a repository that planted its own. `clip` reads the console code page
        # unless the input starts with a UTF-16 byte order mark.
        clip = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', 'clip.exe')
        return ClipboardCommand(argv=(clip,), encoding='utf-16') if os.path.isfile(clip) else None
    else:
        if os.environ.get('WAYLAND_DISPLAY'):
            candidates.append(ClipboardCommand(argv=('wl-copy',)))
        if os.environ.get('DISPLAY'):
            candidates.append(ClipboardCommand(argv=('xclip', '-selection', 'clipboard')))
            candidates.append(ClipboardCommand(argv=('xsel', '--clipboard', '--input')))
    for candidate in candidates:
        if path := shutil.which(candidate.argv[0]):
            return replace(candidate, argv=(path, *candidate.argv[1:]))
    return None


def run_copy(*, command: ClipboardCommand, text: str) -> None:
    """Feed `text` to the clipboard command; a failing one loses only this copy."""
    try:
        # No pipes on the output: `xclip` forks a clipboard owner that would hold them open.
        subprocess.run(
            command.argv,
            input=text.encode(command.encoding),
            env={**os.environ, **command.env},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=COPY_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def copy_text(text: str, *, output: IO[str]) -> None:
    """Put `text` on the clipboard of the machine the user is sitting at.

    Locally, the platform's clipboard command does it in a background thread, so a slow one never
    stalls input. Over SSH, or without such a command, OSC 52 asks the terminal to set its clipboard.
    Not both: some terminals ask before honouring OSC 52, and the local command already did the copy.
    """
    remote = bool(os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_TTY'))
    command = None if remote else copy_command()
    if command is None:
        output.write(make_clipboard_copy(text))
        output.flush()
        return
    threading.Thread(target=run_copy, kwargs={'command': command, 'text': text}, name='clai-copy', daemon=True).start()
