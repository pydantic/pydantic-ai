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
    """Sequence lifecycle entries with consistent middleware ordering."""

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
        """Notify entries from outermost to innermost."""
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
        """Build middleware with the first entry as the outermost layer."""
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
        """Try recovery in the selected direction, forwarding replacement errors."""
        entries = reversed(self.entries) if reverse else iter(self.entries)
        for entry in entries:
            try:
                result = await recover(entry, error)
                if isinstance(result, SkipLifecycleEntry):
                    continue
                return result
            except caught as new_error:
                error = new_error
        raise error

    @asynccontextmanager
    async def stream(
        self,
        source: AsyncIterable[_ValueT],
        wrap: Callable[[_EntryT, AsyncIterable[_ValueT]], AsyncIterable[_ValueT] | None],
    ) -> AsyncGenerator[AsyncIterable[_ValueT]]:
        """Wrap a stream and close every created layer in reverse creation order."""
        stream = source
        streams = [source]
        for entry in reversed(self.entries):
            wrapped = wrap(entry, stream)
            if wrapped is not None:
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
        """Accumulate deferred results while passing only unresolved calls forward."""
        accumulated = DeferredToolResults()
        remaining = requests
        handled = False
        for entry in self.entries:
            result = await handle(entry, remaining)
            if result is None or not (result.approvals or result.calls):
                continue
            handled = True
            accumulated.update(result)
            if (remaining_after_result := remaining.remaining(result)) is None:
                break
            remaining = remaining_after_result
        return accumulated if handled else None
