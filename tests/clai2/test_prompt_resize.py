"""Resize notification lifetime and embedding fallbacks."""

import signal
import sys
from types import FrameType

import pytest

from pydantic_clai2.prompt_resize import resize_notifications

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='SIGWINCH is POSIX-only')


def test_signal_is_chained_and_restored() -> None:
    events: list[str] = []
    previous = signal.getsignal(signal.SIGWINCH)

    def before(signum: int, frame: FrameType | None) -> None:
        events.append('previous')

    signal.signal(signal.SIGWINCH, before)
    try:
        with resize_notifications(lambda: events.append('resize')):
            signal.raise_signal(signal.SIGWINCH)
        assert events == ['resize', 'previous']
        assert signal.getsignal(signal.SIGWINCH) is before
    finally:
        signal.signal(signal.SIGWINCH, previous)


def test_non_main_thread_registration_falls_back_to_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(signum: int, handler: object) -> None:
        raise ValueError('signal only works in main thread')

    monkeypatch.setattr(signal, 'signal', fail)
    with resize_notifications(lambda: None):
        pass


def test_default_signal_disposition_is_restored_without_calling_it() -> None:
    previous = signal.getsignal(signal.SIGWINCH)
    received: list[bool] = []
    signal.signal(signal.SIGWINCH, signal.SIG_DFL)
    try:
        with resize_notifications(lambda: received.append(True)):
            signal.raise_signal(signal.SIGWINCH)
        assert received == [True]
        assert signal.getsignal(signal.SIGWINCH) == signal.SIG_DFL
    finally:
        signal.signal(signal.SIGWINCH, previous)
