"""Bounded background completion work that cannot hold terminal shutdown open."""

import asyncio
from collections.abc import Callable
from contextvars import Context, copy_context
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread

from termflow.tui.completion import Completion  # pyright: ignore[reportMissingTypeStubs]


@dataclass(frozen=True, kw_only=True)
class CompletionRequest:
    """One pure lookup and its owning loop's result future."""

    operation: Callable[[], list[Completion]]
    context: Context
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[list[Completion]]

    def deliver(self, result: list[Completion] | BaseException) -> None:
        """Deliver on the event loop only if the awaiting editor still wants it."""
        if not self.future.done():
            if isinstance(result, BaseException):
                self.future.set_exception(result)
            else:
                self.future.set_result(result)


class CompletionWorker:
    """One daemon thread, one latest queued request, and no terminal ownership.

    Python cannot interrupt an arbitrary synchronous provider. Cancel its await
    and discard its eventual result; never leave a non-daemon pool worker holding
    process exit open. A stuck provider prevents further lookups in this worker,
    not typing, menu handoff, or shutdown.
    """

    def __init__(self) -> None:
        """Start lazily; each editor owns and closes its own worker."""
        self._requests: Queue[CompletionRequest | None] = Queue(maxsize=1)
        self._closed = Event()
        self._thread: Thread | None = None

    async def run(self, operation: Callable[[], list[Completion]]) -> list[Completion]:
        """Queue the newest lookup and await it without shielding cancellation."""
        if self._closed.is_set():
            raise RuntimeError('Completion worker is closed.')
        self._discard_queued()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[Completion]] = loop.create_future()
        self._requests.put_nowait(
            CompletionRequest(operation=operation, context=copy_context(), loop=loop, future=future)
        )
        if self._thread is None:
            self._thread = Thread(target=self._work, name='clai-completion', daemon=True)
            try:
                self._thread.start()
            except BaseException:
                self._thread = None
                self._discard_queued()
                raise
        return await future

    def _discard_queued(self) -> None:
        try:
            previous = self._requests.get_nowait()
        except Empty:
            return
        # Only close enqueues the stop sentinel, and closed workers never drain again.
        assert previous is not None
        previous.future.cancel()

    def close(self) -> None:
        """Stop accepting work and discard queued lookups, without joining a stuck callback."""
        if not self._closed.is_set():
            self._closed.set()
            self._discard_queued()
            self._requests.put_nowait(None)

    def _work(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            try:
                result: list[Completion] | BaseException = request.context.run(request.operation)
            except StopIteration as exc:
                result = RuntimeError('Completion operation raised StopIteration')
                result.__cause__ = exc
            except BaseException as exc:  # noqa: BLE001 -- propagate provider failures to the owning loop.
                result = exc
            if self._closed.is_set():
                return
            try:
                request.loop.call_soon_threadsafe(request.deliver, result)
            except RuntimeError:  # pragma: no cover -- owning loop closed concurrently at process exit.
                return
