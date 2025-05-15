from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from aiolimiter import AsyncLimiter
from tenacity import AsyncRetrying

from . import (
    KnownModelName,
    Model,
    ModelMessage,
    ModelRequestParameters,
    ModelResponse,
    ModelSettings,
    StreamedResponse,
)
from .wrapper import WrapperModel


@dataclass
class RateLimitedModel(WrapperModel):
    """Model which wraps another model such that requests are rate limited with tenacity."""

    def __init__(
        self,
        wrapped: Model | KnownModelName,
        limiter: AsyncLimiter | None = None,
        retryer: AsyncRetrying | None = None,
    ) -> None:
        super().__init__(wrapped)
        self.limiter = limiter
        self.retryer = retryer

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if self.retryer:
            async for attempt in self.retryer:
                with attempt:
                    if self.limiter:
                        async with self.limiter:
                            return await super().request(
                                messages,
                                model_settings,
                                model_request_parameters,
                            )
                    else:
                        return await super().request(
                            messages,
                            model_settings,
                            model_request_parameters,
                        )
            raise RuntimeError('Model request failed after all retries')
        else:
            if self.limiter:
                async with self.limiter:
                    return await super().request(
                        messages,
                        model_settings,
                        model_request_parameters,
                    )
            else:
                return await super().request(
                    messages,
                    model_settings,
                    model_request_parameters,
                )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        if self.retryer:
            async for attempt in self.retryer:
                with attempt:
                    if self.limiter:
                        async with self.limiter:
                            async with super().request_stream(
                                messages, model_settings, model_request_parameters
                            ) as response_stream:
                                yield response_stream
                    else:
                        async with super().request_stream(
                            messages, model_settings, model_request_parameters
                        ) as response_stream:
                            yield response_stream
        else:
            if self.limiter:
                async with self.limiter:
                    async with super().request_stream(
                        messages, model_settings, model_request_parameters
                    ) as response_stream:
                        yield response_stream
            else:
                async with super().request_stream(
                    messages, model_settings, model_request_parameters
                ) as response_stream:
                    yield response_stream
