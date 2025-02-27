"""
This module provides an asynchronous context manager and iterator,
`MockAsyncStream`, which wraps a synchronous iterator and exposes it
as an asynchronous stream. It can be used in async/await code blocks
and async for-loops, emulating asynchronous iteration behavior.
"""

from __future__ import annotations as _annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic_ai import _utils

from ..conftest import raise_if_exception

T = TypeVar('T')


@dataclass
class MockAsyncStream(Generic[T]):
    """Wraps a synchronous iterator in an asynchronous interface.

    This class allows a synchronous iterator to be treated as an
    asynchronous iterator, enabling iteration in `async for` loops
    and usage within `async with` blocks.

    Example usage:
        async def example():
            sync_iter = iter([1, 2, 3])
            async_stream = MockAsyncStream(sync_iter)

            async for item in async_stream:
                print(item)

            async with MockAsyncStream(sync_iter) as stream:
                async for item in stream:
                    print(item)
    """

    _iter: Iterator[T]
    """The underlying synchronous iterator."""

    async def __anext__(self) -> T:
        """Return the next item from the synchronous iterator as if it were asynchronous.

        Calls `_utils.sync_anext` to retrieve the next item from the underlying
        synchronous iterator. If the iterator is exhausted, `StopAsyncIteration`
        is raised.
        """
        next = _utils.sync_anext(self._iter)
        raise_if_exception(next)
        return next

    def __aiter__(self) -> MockAsyncStream[T]:
        return self

    async def __aenter__(self) -> MockAsyncStream[T]:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        pass
