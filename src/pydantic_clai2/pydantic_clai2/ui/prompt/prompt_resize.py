"""Scope resize notifications without doing terminal IO from a signal handler."""

import signal
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from types import FrameType


@contextmanager
def resize_notifications(notify: Callable[[], None]) -> Generator[None]:
    """Chain and restore SIGWINCH; unsupported hosts retain size polling."""
    if sys.platform == 'win32':  # pragma: no cover -- POSIX signal is unavailable on Windows.
        yield
        return
    previous = signal.getsignal(signal.SIGWINCH)

    def changed(signum: int, frame: FrameType | None) -> None:
        notify()
        if callable(previous):
            previous(signum, frame)

    try:
        signal.signal(signal.SIGWINCH, changed)
    except ValueError:  # Embedded shells may run outside the main thread.
        yield
    else:
        try:
            yield
        finally:
            signal.signal(signal.SIGWINCH, previous)
