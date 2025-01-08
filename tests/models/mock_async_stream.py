from __future__ import annotations as _annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic_ai import _utils

T = TypeVar('T')


@dataclass
class MockAsyncStream(Generic[T]):
    _iter: Iterator[T]

    async def __anext__(self) -> T:
        return _utils.sync_anext(self._iter)

    def __aiter__(self) -> MockAsyncStream[T]:
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: Any):
        pass
