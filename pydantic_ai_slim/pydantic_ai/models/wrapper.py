from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from ..messages import ModelMessage, ModelResponse
from ..settings import ModelSettings
from ..usage import Usage
from . import Model, ModelRequestParameters, StreamedResponse


@dataclass
class WrapperModel(Model):
    """Model which wraps another model."""

    wrapped: Model

    async def request(self, *args: Any, **kwargs: Any) -> tuple[ModelResponse, Usage]:
        return await self.wrapped.request(*args, **kwargs)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        async with self.wrapped.request_stream(messages, model_settings, model_request_parameters) as response_stream:
            yield response_stream

    @property
    def model_name(self) -> str:
        return self.wrapped.model_name

    @property
    def system(self) -> str | None:
        return self.wrapped.system

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)
