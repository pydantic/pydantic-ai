"""Separate terminal interrupts from cancellation of the application task."""

import asyncio
import signal
import threading
import time
from collections.abc import Awaitable, Callable
from types import FrameType


class Interrupts:
    """Cancel one operation, or request exit on a second press within two seconds."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Use a monotonic clock so wall-clock changes cannot alter the window."""
        self._clock = clock
        self._last: float | None = None
        self.exit_requested = False
        self._cancel: Callable[[bool], None] | None = None

    def press(self) -> bool:
        """Share the double-press window between running and input modes."""
        now = self._clock()
        self.exit_requested = self._last is not None and now - self._last <= 2
        self._last = now
        return self.exit_requested

    @property
    def active(self) -> bool:
        """Whether an operation currently accepts editor cancellation."""
        return self._cancel is not None

    def cancel(self, *, exit_on_repeat: bool = True) -> bool:
        """Interrupt active work from an editor key without sending a process signal."""
        if self._cancel is None:
            return False
        self._cancel(exit_on_repeat)
        return True

    async def run(self, operation: Awaitable[None]) -> bool:
        """Return false for user cancellation; propagate external task cancellation."""
        main_thread = threading.current_thread() is threading.main_thread()

        async def invoke() -> None:
            await operation

        task = asyncio.create_task(invoke())
        interrupted = False

        def cancel(exit_on_repeat: bool) -> None:
            nonlocal interrupted
            if exit_on_repeat:
                self.press()
            if not interrupted:
                interrupted = True
                task.cancel()

        def on_signal(signum: int, frame: FrameType | None) -> None:
            cancel(True)

        previous = signal.getsignal(signal.SIGINT)
        self._cancel = cancel
        try:
            if main_thread:
                signal.signal(signal.SIGINT, on_signal)
            try:
                await task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if not interrupted or current is not None and current.cancelling():
                    raise
                return False
            return True
        finally:
            self._cancel = None
            if main_thread:
                signal.signal(signal.SIGINT, previous)
