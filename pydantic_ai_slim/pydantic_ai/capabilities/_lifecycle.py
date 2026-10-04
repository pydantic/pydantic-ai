from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Generic, ParamSpec, TypeVar

from pydantic_ai import _utils
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

_EntryT = TypeVar('_EntryT')
_ValueT = TypeVar('_ValueT')
_ResultT = TypeVar('_ResultT')
_ErrorT = TypeVar('_ErrorT', bound=BaseException)
_Params = ParamSpec('_Params')


@dataclass(frozen=True)
class SkipLifecycleEntry:
    """Signal that a lifecycle entry is inactive for the current invocation."""


SKIP_LIFECYCLE_ENTRY = SkipLifecycleEntry()


@dataclass(frozen=True)
class LifecycleStack(Generic[_EntryT]):
    """Sequence lifecycle entries with consistent middleware ordering.

    The first entry is the outermost layer. Callers own what an entry is, whether it is active, and how
    it is invoked; the stack owns only the order and the error, stream and deferred-result bookkeeping.
    """

    entries: Sequence[_EntryT]

    async def forward(
        self,
        value: _ValueT,
        transform: Callable[[_EntryT, _ValueT], Awaitable[_ValueT]],
    ) -> _ValueT:
        """Transform a value from the outermost entry to the innermost."""
        for entry in self.entries:
            value = await transform(entry, value)
        return value

    async def reverse(
        self,
        value: _ValueT,
        transform: Callable[[_EntryT, _ValueT], Awaitable[_ValueT]],
    ) -> _ValueT:
        """Transform a value from the innermost entry to the outermost."""
        for entry in reversed(self.entries):
            value = await transform(entry, value)
        return value

    async def notify(self, notify: Callable[[_EntryT], Awaitable[None]]) -> None:
        """Notify entries from the outermost to the innermost."""
        for entry in self.entries:
            await notify(entry)

    def wrap(
        self,
        handler: Callable[_Params, Awaitable[_ResultT]],
        wrap: Callable[
            [_EntryT, Callable[_Params, Awaitable[_ResultT]]],
            Callable[_Params, Awaitable[_ResultT]],
        ],
    ) -> Callable[_Params, Awaitable[_ResultT]]:
        """Build a middleware chain with the first entry as the outermost layer."""
        for entry in reversed(self.entries):
            handler = wrap(entry, handler)
        return handler

    async def recover(
        self,
        error: _ErrorT,
        recover: Callable[[_EntryT, _ErrorT], Awaitable[_ResultT | SkipLifecycleEntry]],
        caught: type[_ErrorT] | tuple[type[_ErrorT], ...],
        *,
        reverse: bool = True,
    ) -> _ResultT:
        """Offer the error to each entry until one recovers, passing replacement errors on.

        An inactive entry returns `SKIP_LIFECYCLE_ENTRY` rather than re-raising, so skipping it leaves
        the propagating error's traceback untouched.
        """
        entries = reversed(self.entries) if reverse else iter(self.entries)
        for entry in entries:
            try:
                result = await recover(entry, error)
            except caught as new_error:
                error = new_error
                continue
            if isinstance(result, SkipLifecycleEntry):
                continue
            return result
        raise error

    @asynccontextmanager
    async def stream(
        self,
        source: AsyncIterable[_ValueT],
        wrap: Callable[[_EntryT, AsyncIterable[_ValueT]], AsyncIterable[_ValueT] | None],
    ) -> AsyncGenerator[AsyncIterable[_ValueT]]:
        """Wrap a stream with the first entry outermost, and close every layer innermost-last on exit.

        An entry that returns `None` adds no layer.
        """
        stream = source
        streams = [source]
        for entry in reversed(self.entries):
            if (wrapped := wrap(entry, stream)) is not None:
                stream = wrapped
                streams.append(stream)
        try:
            yield stream
        finally:
            await _utils.aclose_all(reversed(streams))

    async def settle_deferred(
        self,
        requests: DeferredToolRequests,
        handle: Callable[[_EntryT, DeferredToolRequests], Awaitable[DeferredToolResults | None]],
    ) -> DeferredToolResults | None:
        """Accumulate deferred results, passing each entry only the calls still unresolved."""
        accumulated = DeferredToolResults()
        remaining = requests
        handled = False
        for entry in self.entries:
            result = await handle(entry, remaining)
            if result is None or not (result.approvals or result.calls):
                continue
            handled = True
            accumulated.update(result)
            if (still_remaining := remaining.remaining(result)) is None:
                break
            remaining = still_remaining
        return accumulated if handled else None
