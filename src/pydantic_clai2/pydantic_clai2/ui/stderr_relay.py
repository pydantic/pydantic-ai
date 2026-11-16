"""Keep macOS's `MallocStackLogging` fork notices off the interactive terminal.

Once a process receives a memory pressure event, macOS's allocator loads its stack logging library. From then on
the child half of every `fork()` reports `python(PID) MallocStackLogging: can't turn off malloc stack logging
because it was not enabled.` on standard error, before the child sets up its own streams and starts the program.
That standard error is still CLAI's, the terminal, so every subprocess (a shell tool call, `git`, a notification)
prints a line through the live display. No environment variable is involved, and a child's own `stderr=PIPE`
cannot catch it.

While the interactive session runs on macOS with standard error on a terminal, file descriptor 2 is a pipe
instead. A thread copies everything written to it to the terminal unchanged, except those exact lines.
"""

import os
import re
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager

_NOTICE = re.compile(
    rb"[^\s()]+\(\d+\) MallocStackLogging: can't turn off malloc stack logging because it was not enabled\.\r?\n"
)
# One `os.read` gets everything in the pipe up to this size; the allocator writes each notice in a single `write`.
_CHUNK = 65536
# How long leaving the session waits for the copy to reach the terminal.
_DRAIN_TIMEOUT = 1.0

_terminal: int | None = None


def drop_fork_notices(data: bytes, *, at_line_start: bool = True) -> bytes:
    """`data` without the whole `MallocStackLogging` fork notice lines in it.

    `at_line_start` is whether `data` starts a line, rather than continuing one an earlier write left open; only
    whole lines are notices.
    """
    lines = data.splitlines(keepends=True)
    return b''.join(
        line for index, line in enumerate(lines) if (index == 0 and not at_line_start) or not _NOTICE.fullmatch(line)
    )


def terminal_stderr() -> int | None:
    """The terminal's standard error while `relay_stderr` holds file descriptor 2, else `None`.

    Pass it as a subprocess's `stderr` when the child must draw on the terminal itself, such as a progress bar.
    `None` keeps the default of inheriting CLAI's standard error.
    """
    return _terminal


def _copy(source: int, terminal: int) -> None:
    at_line_start = True
    try:
        while data := os.read(source, _CHUNK):
            view = memoryview(drop_fork_notices(data, at_line_start=at_line_start))
            at_line_start = data.endswith((b'\n', b'\r'))
            while view:
                view = view[os.write(terminal, view) :]
    finally:
        os.close(source)
        os.close(terminal)


@contextmanager
def relay_stderr() -> Generator[None]:
    """Copy standard error to the terminal without `MallocStackLogging` fork notices, on macOS.

    Elsewhere, or when standard error is not a terminal, this does nothing. Children started meanwhile inherit the
    pipe, so what they write reaches the terminal the same way; one that outlives the session keeps the copy going.
    """
    global _terminal
    if sys.platform != 'darwin' or not os.isatty(2):
        yield
        return
    sys.stderr.flush()
    terminal = os.dup(2)
    source, sink = os.pipe()
    os.dup2(sink, 2)
    os.close(sink)
    copier = threading.Thread(target=_copy, args=(source, os.dup(terminal)), name='clai-stderr', daemon=True)
    copier.start()
    _terminal = terminal
    try:
        yield
    finally:
        _terminal = None
        sys.stderr.flush()
        os.dup2(terminal, 2)
        os.close(terminal)
        copier.join(_DRAIN_TIMEOUT)
