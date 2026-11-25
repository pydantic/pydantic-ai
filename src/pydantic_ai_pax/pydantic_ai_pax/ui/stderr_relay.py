"""Own the terminal's standard error: off it during the interactive session, and without macOS fork notices around it.

While the interactive session runs on a POSIX terminal, file descriptor 2 is a pipe (`capture_stderr`). Pax's own
writes, `logging.lastResort`, warnings, a library's handler on `sys.stderr`, and every child started meanwhile write
to it, and a thread appends what arrives to the session's log, so nothing paints over the live display. A child that
must draw on the terminal itself, such as an editor, passes `terminal_stderr()` as its `stderr`.

Once a process receives a memory pressure event, macOS's allocator loads its stack logging library. From then on
the child half of every `fork()` reports `python(PID) MallocStackLogging: can't turn off malloc stack logging
because it was not enabled.` on standard error, before the child sets up its own streams and starts the program.
No environment variable is involved, and a child's own `stderr=PIPE` cannot catch it. The log copy drops those
exact lines. Around the session, as for the `git` calls the CLI makes, `relay_stderr` holds file descriptor 2 as a
pipe on macOS and copies everything else written to it to the terminal unchanged.
"""

import codecs
import os
import re
import sys
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager

_NOTICE = re.compile(
    rb"[^\s()]+\(\d+\) MallocStackLogging: can't turn off malloc stack logging because it was not enabled\.\r?\n"
)
# One `os.read` gets everything in the pipe up to this size; the allocator writes each notice in a single `write`.
_CHUNK = 65536
# How long leaving the session, or the relay around it, waits for the copy to reach the log or the terminal.
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
    """The terminal's standard error while `capture_stderr` or `relay_stderr` holds file descriptor 2, else `None`.

    Pass it as a subprocess's `stderr` when the child must draw on the terminal itself, such as a progress bar.
    `None` keeps the default of inheriting Pax's standard error.
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


def _copy_to(source: int, append: Callable[[str], None]) -> None:
    at_line_start = True
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    try:
        while data := os.read(source, _CHUNK):
            kept = drop_fork_notices(data, at_line_start=at_line_start)
            at_line_start = data.endswith((b'\n', b'\r'))
            if text := decoder.decode(kept):
                append(text)
        if text := decoder.decode(b'', final=True):
            append(text)
    finally:
        os.close(source)


@contextmanager
def capture_stderr(append: Callable[[str], None]) -> Generator[None]:
    """Hand everything written to standard error to `append` as text, while it is a POSIX terminal.

    File descriptor 2 is a pipe for the block, so children started meanwhile write to it too; a thread decodes what
    arrives, without `MallocStackLogging` fork notices. Elsewhere, or when standard error is not a terminal, this
    does nothing. Under `relay_stderr`, the terminal is the one it holds.
    """
    global _terminal
    shown = _terminal
    if sys.platform == 'win32' or not os.isatty(2 if shown is None else shown):
        yield
        return
    sys.stderr.flush()
    restore = os.dup(2)
    terminal = os.dup(2 if shown is None else shown)
    source, sink = os.pipe()
    os.dup2(sink, 2)
    os.close(sink)
    copier = threading.Thread(target=_copy_to, args=(source, append), name='pax-stderr-log', daemon=True)
    copier.start()
    _terminal = terminal
    try:
        yield
    finally:
        _terminal = shown
        sys.stderr.flush()
        os.dup2(restore, 2)
        os.close(restore)
        os.close(terminal)
        copier.join(_DRAIN_TIMEOUT)


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
    copier = threading.Thread(target=_copy, args=(source, os.dup(terminal)), name='pax-stderr', daemon=True)
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
