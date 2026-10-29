"""Separate terminal interrupts from cancellation of the application task."""

import asyncio
import signal
import socket
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import FrameType
from typing import TypeAlias


class Interrupts:
    """Cancel one operation, or request exit on a second press within two seconds."""

    def __init__(self, *, signals: bool = True, clock: Callable[[], float] = time.monotonic) -> None:
        """Use a monotonic clock so wall-clock changes cannot alter the window.

        With `signals`, as for the session the terminal shows, Ctrl-C cancels the operation `run` awaits.
        Other sessions leave the process's Ctrl-C handler alone, so overlapping runs cannot restore each
        other's, and are cancelled with `cancel` only.
        """
        self.signals = signals
        self._clock = clock
        self._last: float | None = None
        self.exit_requested = False
        self._cancel: Callable[[bool], None] | None = None
        self._handler: _Handler | None = None
        """The Ctrl-C handler of the operation `run` awaits now."""

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

    def focus(self, *, signals: bool) -> None:
        """Take Ctrl-C, or give it up, also for the operation `run` awaits now: the terminal moved to or from it."""
        self.signals = signals
        if self._handler is not None:
            self._take(self._handler) if signals else self._give(self._handler)

    async def run(self, operation: Awaitable[None]) -> bool:
        """Return false for user cancellation; propagate external task cancellation."""
        loop = asyncio.get_running_loop()

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
            # A Python signal handler can run between any two bytecodes, including inside asyncio's
            # own callbacks, such as `asyncio.shield` checking and then setting its result.
            # Cancelling from the loop keeps those steps whole and also wakes an idle selector.
            loop.call_soon_threadsafe(cancel, True)

        outer, self._handler = self._handler, _Handler(handle=on_signal, loop=loop)
        handler = self._handler
        self._cancel = cancel
        try:
            self._take(handler)
            try:
                # `asyncio.wait` never forwards our own cancellation to `task`, so a
                # `CancelledError` here is always external and must propagate.
                await asyncio.wait({task})
            except asyncio.CancelledError:
                task.cancel()
                await asyncio.wait({task})
                raise
            if interrupted and task.cancelled():
                return False
            await task  # Re-raise the operation's failure, or a cancellation that was not ours.
            return True
        finally:
            self._cancel = None
            self._give(handler)
            self._handler = outer

    def _take(self, handler: '_Handler') -> None:
        global _stop_waking
        if (
            self.signals
            and threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGINT) is not handler.handle
        ):
            # Taken before and replaced since: leave that place in the chain before taking the top.
            self._give(handler)
            if not _TAKEN:
                _stop_waking = _wake(handler.loop)
            handler.previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, handler.handle)
            _TAKEN.append(handler)

    def _give(self, handler: '_Handler') -> None:
        global _stop_waking
        if handler not in _TAKEN:
            return
        _TAKEN.remove(handler)
        if signal.getsignal(signal.SIGINT) is handler.handle:
            signal.signal(signal.SIGINT, handler.previous)
        else:
            # Another took Ctrl-C over this one, which ends first: that one gives it back to what this one replaced.
            for other in _TAKEN:
                if other.previous is handler.handle:
                    other.previous = handler.previous
        if not _TAKEN:
            stop, _stop_waking = _stop_waking, _no_wakeup
            stop()


@dataclass(kw_only=True, eq=False)
class _Handler:
    handle: Callable[[int, FrameType | None], None]
    loop: asyncio.AbstractEventLoop
    """The loop of the operation `handle` cancels."""
    previous: '_SignalHandler' = signal.SIG_DFL
    """What Ctrl-C did before `handle` took it."""


def _wake(loop: asyncio.AbstractEventLoop) -> Callable[[], None]:
    """Make a signal wake `loop` until the returned function is called.

    Python runs a signal handler only between bytecodes, so Ctrl-C landing as the loop blocks in its
    selector, or on another thread, would wait for the loop's next event, such as a quiet command
    ending. A wakeup fd lets the signal itself wake the selector. One already set, such as the one
    asyncio sets for `add_signal_handler` or the Windows proactor's, wakes the loop instead, for as
    long as its owner keeps it.
    """
    reader, writer = socket.socketpair()
    reader.setblocking(False)
    writer.setblocking(False)
    if (previous := signal.set_wakeup_fd(writer.fileno(), warn_on_full_buffer=False)) != -1:
        signal.set_wakeup_fd(previous)
        reader.close()
        writer.close()
        return _no_wakeup
    loop.add_reader(reader, reader.recv, 4096)

    def stop() -> None:
        if (current := signal.set_wakeup_fd(-1)) != writer.fileno():
            # Replaced since, as by `add_signal_handler`: that one is its setter's to clear.
            signal.set_wakeup_fd(current)
        loop.remove_reader(reader)
        reader.close()
        writer.close()

    return stop


_SignalHandler: TypeAlias = Callable[[int, FrameType | None], object] | int | signal.Handlers | None

_TAKEN: list[_Handler] = []
"""The handlers holding Ctrl-C, oldest first, for every `Interrupts`: the process has one SIGINT handler."""


def _no_wakeup() -> None:
    """Undo nothing: `_wake` set no wakeup fd of its own."""


_stop_waking: Callable[[], None] = _no_wakeup
"""Undoes the `_wake` of the loop of the handlers in `_TAKEN`, which all run on the main thread's loop."""
