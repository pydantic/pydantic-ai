from __future__ import annotations as _annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, is_dataclass
from functools import partial
from types import GenericAlias
from typing import Any, Callable, Generic, TypeVar, Union, cast, overload

from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import ParamSpec, TypeAlias, is_typeddict

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


# With PEP-728 this should be a TypedDict with `type: Literal['object']`, and `extra_items=Any`
ObjectJsonSchema: TypeAlias = dict[str, Any]


def check_object_json_schema(schema: JsonSchemaValue) -> ObjectJsonSchema:
    from .exceptions import UserError

    if schema.get('type') == 'object':
        return schema
    else:
        raise UserError('Schema must be an object')


T = TypeVar('T')


@dataclass
class Some(Generic[T]):
    """Analogous to Rust's `Option::Some` type."""

    value: T


Option: TypeAlias = Union[Some[T], None]
"""Analogous to Rust's `Option` type, usage: `Option[Thing]` is equivalent to `Some[Thing] | None`."""


Left = TypeVar('Left')
Right = TypeVar('Right')


class Unset:
    """A singleton to represent an unset value."""

    pass


UNSET = Unset()


class Either(Generic[Left, Right]):
    """Two member Union that records which member was set, this is analogous to Rust enums with two variants.

    Usage:

    ```py
    if left_thing := either.left:
        use_left(left_thing.value)
    else:
        use_right(either.right)
    ```
    """

    __slots__ = '_left', '_right'

    @overload
    def __init__(self, *, left: Left) -> None: ...

    @overload
    def __init__(self, *, right: Right) -> None: ...

    def __init__(self, left: Left | Unset = UNSET, right: Right | Unset = UNSET) -> None:
        if left is not UNSET:
            assert right is UNSET, '`Either` must receive exactly one argument - `left` or `right`'
            self._left: Option[Left] = Some(cast(Left, left))
        else:
            assert right is not UNSET, '`Either` must receive exactly one argument - `left` or `right`'
            self._left = None
            self._right = cast(Right, right)

    @property
    def left(self) -> Option[Left]:
        return self._left

    @property
    def right(self) -> Right:
        return self._right

    def is_left(self) -> bool:
        return self._left is not None

    def whichever(self) -> Left | Right:
        return self._left.value if self._left is not None else self.right


async def group_by_temporal(aiter: AsyncIterator[T], soft_max_interval: float | None) -> AsyncIterator[list[T]]:
    """Group items from an async iterable into lists based on time interval between them.

    Effectively debouncing the iterator.

    Args:
        aiter: The async iterable to group.
        soft_max_interval: Maximum interval over which to group items, this should avoid a trickle of items causing
            a group to never be yielded. It's a soft max in the sense that once we're over this time, we yield items
            as soon as `aiter.__anext__()` returns. If `None`, no grouping/debouncing is performed

    Returns: An async iterable of lists of items from the input async iterable.
    """
    if soft_max_interval is None:
        async for item in aiter:
            yield [item]
        return

    assert soft_max_interval is not None and soft_max_interval >= 0, 'soft_max_interval must be a positive number'
    buffer: list[T] = []
    group_start_time = time.monotonic()
    # we might wait for the next item more than once, so we store the coros to await next time if any
    coro: asyncio.Task[T] | None = None

    while True:
        if group_start_time is None:
            # group hasn't started, we just wait for the maximum interval
            wait_time = soft_max_interval
        else:
            # wait for the time remaining in the group
            wait_time = soft_max_interval - (time.monotonic() - group_start_time)

        # if there's no current coroutine, we get the next one
        if coro is None:
            # aiter.__anext__() returns an Awaitable[T], not a Coroutine which asyncio.create_task expects
            # TODO does this matter? It seems to run fine
            coro = asyncio.create_task(aiter.__anext__())  # pyright: ignore[reportArgumentType]

        # we use asyncio.wait to avoid cancelling the coroutine if it's not done
        done, _ = await asyncio.wait((coro,), timeout=wait_time)

        if done:
            # the one task we waited for completed
            try:
                item = done.pop().result()
            except StopAsyncIteration:
                # if the coro raised StopAsyncIteration, we're done iterating
                if buffer:
                    yield buffer
                break
            else:
                # we got an item, add it to the buffer and set coro to None to get the next item
                buffer.append(item)
                coro = None
                # if this is the first item in the group, set the group start time
                if group_start_time is None:
                    group_start_time = time.monotonic()
        elif buffer:
            yield buffer
            buffer = []
            group_start_time = None
