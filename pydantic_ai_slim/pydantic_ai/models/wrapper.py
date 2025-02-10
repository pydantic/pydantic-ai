from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from ..messages import ModelResponse
from ..usage import Usage
from . import Model, StreamedResponse


@dataclass
class WrapperModel(Model):
    """Model which wraps another model."""

    wrapped: Model

    async def request(self, *args: Any, **kwargs: Any) -> tuple[ModelResponse, Usage]:
        return await self.wrapped.request(*args, **kwargs)

    @asynccontextmanager
    async def request_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[StreamedResponse]:
        async with self.wrapped.request_stream(*args, **kwargs) as stream:
            yield stream

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)
