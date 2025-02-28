from __future__ import annotations as _annotations

import asyncio
import time
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, is_dataclass
from datetime import datetime, timezone
from functools import partial
from types import GenericAlias
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, Union

from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import ParamSpec, TypeAlias, TypeGuard, is_typeddict

if TYPE_CHECKING:
    from . import messages as _messages
    from .tools import ObjectJsonSchema

_P = ParamSpec('_P')
_R = TypeVar('_R')


async def run_in_executor(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    if kwargs:
        # noinspection PyTypeChecker
        return await asyncio.get_running_loop().run_in_executor(None, partial(func, *args, **kwargs))
    else:
        return await asyncio.get_running_loop().run_in_executor(None, func, *args)  # type: ignore


def is_model_like(type_: Any) -> bool:
    """Check if something is a pydantic model, dataclass or typedict.

    These should all generate a JSON Schema with `{"type": "object"}` and therefore be usable directly as
    function parameters.
    """
    return (
        isinstance(type_, type)
        and not isinstance(type_, GenericAlias)
        and (issubclass(type_, BaseModel) or is_dataclass(type_) or is_typeddict(type_))
    )


def check_object_json_schema(schema: JsonSchemaValue) -> ObjectJsonSchema:
    from .exceptions import UserError

    if schema.get('type') == 'object':
        return schema
    elif schema.get('$ref') is not None:
        return schema.get('$defs', {}).get(schema['$ref'][8:])  # This removes the initial "#/$defs/".
    else:
        raise UserError('Schema must be an object')


T = TypeVar('T')


@dataclass
class Some(Generic[T]):
    """Analogous to Rust's `Option::Some` type."""

    value: T


Option: TypeAlias = Union[Some[T], None]
"""Analogous to Rust's `Option` type, usage: `Option[Thing]` is equivalent to `Some[Thing] | None`."""


class Unset:
    """A singleton to represent an unset value."""

    pass


UNSET = Unset()


def is_set(t_or_unset: T | Unset) -> TypeGuard[T]:
    return t_or_unset is not UNSET


@asynccontextmanager
async def group_by_temporal(
    aiterable: AsyncIterable[T], soft_max_interval: float | None
) -> AsyncIterator[AsyncIterable[list[T]]]:
    """Group items from an async iterable into lists based on time interval between them.

    Effectively, this debounces the iterator.

    This returns a context manager usable as an iterator so any pending tasks can be cancelled if an error occurs
    during iteration.

    Usage:

    ```python
    async with group_by_temporal(yield_groups(), 0.1) as groups_iter:
        async for groups in groups_iter:
            print(groups)
    ```

    Args:
        aiterable: The async iterable to group.
        soft_max_interval: Maximum interval over which to group items, this should avoid a trickle of items causing
            a group to never be yielded. It's a soft max in the sense that once we're over this time, we yield items
            as soon as `aiter.__anext__()` returns. If `None`, no grouping/debouncing is performed

    Returns:
        A context manager usable as an async iterable of lists of items produced by the input async iterable.
    """
    if soft_max_interval is None:

        async def async_iter_groups_noop() -> AsyncIterator[list[T]]:
            async for item in aiterable:
                yield [item]

        yield async_iter_groups_noop()
        return

    # we might wait for the next item more than once, so we store the task to await next time
    task: asyncio.Task[T] | None = None

    async def async_iter_groups() -> AsyncIterator[list[T]]:
        nonlocal task

        assert soft_max_interval is not None and soft_max_interval >= 0, 'soft_max_interval must be a positive number'
        buffer: list[T] = []
        group_start_time = time.monotonic()

        aiterator = aiterable.__aiter__()
        while True:
            if group_start_time is None:
                # group hasn't started, we just wait for the maximum interval
                wait_time = soft_max_interval
            else:
                # wait for the time remaining in the group
                wait_time = soft_max_interval - (time.monotonic() - group_start_time)

            # if there's no current task, we get the next one
            if task is None:
                # aiter.__anext__() returns an Awaitable[T], not a Coroutine which asyncio.create_task expects
                # so far, this doesn't seem to be a problem
                task = asyncio.create_task(aiterator.__anext__())  # pyright: ignore[reportArgumentType]

            # we use asyncio.wait to avoid cancelling the coroutine if it's not done
            done, _ = await asyncio.wait((task,), timeout=wait_time)

            if done:
                # the one task we waited for completed
                try:
                    item = done.pop().result()
                except StopAsyncIteration:
                    # if the task raised StopAsyncIteration, we're done iterating
                    if buffer:
                        yield buffer
                    task = None
                    break
                else:
                    # we got an item, add it to the buffer and set task to None to get the next item
                    buffer.append(item)
                    task = None
                    # if this is the first item in the group, set the group start time
                    if group_start_time is None:
                        group_start_time = time.monotonic()
            elif buffer:
                # otherwise if the task timeout expired and we have items in the buffer, yield the buffer
                yield buffer
                # clear the buffer and reset the group start time ready for the next group
                buffer = []
                group_start_time = None

    try:
        yield async_iter_groups()
    finally:  # pragma: no cover
        # after iteration if a tasks still exists, cancel it, this will only happen if an error occurred
        if task:
            task.cancel('Cancelling due to error in iterator')
            with suppress(asyncio.CancelledError):
                await task


def sync_anext(iterator: Iterator[T]) -> T:
    """Get the next item from a sync iterator, raising `StopAsyncIteration` if it's exhausted.

    Useful when iterating over a sync iterator in an async context.
    """
    try:
        return next(iterator)
    except StopIteration as e:
        raise StopAsyncIteration() from e


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def guard_tool_call_id(
    t: _messages.ToolCallPart | _messages.ToolReturnPart | _messages.RetryPromptPart, model_source: str
) -> str:
    """Type guard that checks a `tool_call_id` is not None both for static typing and runtime."""
    assert t.tool_call_id is not None, f'{model_source} requires `tool_call_id` to be set: {t}'
    return t.tool_call_id


class PeekableAsyncStream(Generic[T]):
    """Wraps an async iterable of type T and allows peeking at the *next* item without consuming it.

    We only buffer one item at a time (the next item). Once that item is yielded, it is discarded.
    This is a single-pass stream.
    """

    def __init__(self, source: AsyncIterable[T]):
        self._source = source
        self._source_iter: AsyncIterator[T] | None = None
        self._buffer: T | Unset = UNSET
        self._exhausted = False

    async def peek(self) -> T | Unset:
        """Returns the next item that would be yielded without consuming it.

        Returns None if the stream is exhausted.
        """
        if self._exhausted:
            return UNSET

        # If we already have a buffered item, just return it.
        if not isinstance(self._buffer, Unset):
            return self._buffer

        # Otherwise, we need to fetch the next item from the underlying iterator.
        if self._source_iter is None:
            self._source_iter = self._source.__aiter__()

        try:
            self._buffer = await self._source_iter.__anext__()
        except StopAsyncIteration:
            self._exhausted = True
            return UNSET

        return self._buffer

    async def is_exhausted(self) -> bool:
        """Returns True if the stream is exhausted, False otherwise."""
        return isinstance(await self.peek(), Unset)

    def __aiter__(self) -> AsyncIterator[T]:
        # For a single-pass iteration, we can return self as the iterator.
        return self

    async def __anext__(self) -> T:
        """Yields the buffered item if present, otherwise fetches the next item from the underlying source.

        Raises StopAsyncIteration if the stream is exhausted.
        """
        if self._exhausted:
            raise StopAsyncIteration

        # If we have a buffered item, yield it.
        if not isinstance(self._buffer, Unset):
            item = self._buffer
            self._buffer = UNSET
            return item

        # Otherwise, fetch the next item from the source.
        if self._source_iter is None:
            self._source_iter = self._source.__aiter__()

        try:
            return await self._source_iter.__anext__()
        except StopAsyncIteration:
            self._exhausted = True
            raise
