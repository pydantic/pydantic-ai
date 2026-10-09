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
    """A local command that sets the system clipboard from UTF-8 on its standard input."""

    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict[str, str])
    """Variables set for the command on top of CLAI's own environment."""


def copy_command(*, platform: str = sys.platform) -> ClipboardCommand | None:
    """The installed clipboard command for this machine, or `None` when there is none."""
    candidates: list[ClipboardCommand] = []
    if platform == 'darwin':
        # Without a UTF-8 locale, `pbcopy` reads its input as Mac Roman.
        candidates.append(ClipboardCommand(argv=('pbcopy',), env={'LC_CTYPE': 'UTF-8'}))
    elif platform == 'win32':
        # The system's own PowerShell: a `PATH` search on Windows looks in the working directory
        # first, which may be a repository that planted its own. Not `clip.exe`, which reads the
        # console code page and keeps the byte order mark that would make it read UTF-16.
        system = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')
        powershell = os.path.join(system, 'WindowsPowerShell', 'v1.0', 'powershell.exe')
        script = '[Console]::InputEncoding = [Text.Encoding]::UTF8; Set-Clipboard -Value ([Console]::In.ReadToEnd())'
        argv = (powershell, '-NoProfile', '-NonInteractive', '-Command', script)
        return ClipboardCommand(argv=argv) if os.path.isfile(powershell) else None
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
            input=text.encode(),
            env={**os.environ, **command.env},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=COPY_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


class LatestCopy:
    """Run one clipboard command at a time, and only the newest copy that is waiting.

    Copies made while one runs replace each other, so the clipboard ends with the last selection
    even when a slow command finishes late, and quick drags never pile up threads or processes.
    """

    def __init__(self) -> None:
        """Start idle."""
        self._lock = threading.Lock()
        self._pending: tuple[ClipboardCommand, str] | None = None
        self._running = False

    def submit(self, *, command: ClipboardCommand, text: str) -> None:
        """Copy `text` after the running copy, replacing any copy still waiting."""
        with self._lock:
            self._pending = (command, text)
            if self._running:
                return
            self._running = True
        # Not a daemon: exiting right after a copy waits for it, at most `COPY_TIMEOUT`, rather than losing it.
        threading.Thread(target=self._drain, name='clai-copy', daemon=False).start()

    def _drain(self) -> None:
        while True:
            with self._lock:
                job, self._pending = self._pending, None
                if job is None:
                    self._running = False
                    return
            command, text = job
            run_copy(command=command, text=text)


_COPIES = LatestCopy()


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
    _COPIES.submit(command=command, text=text)
