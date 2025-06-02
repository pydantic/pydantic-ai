from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tenacity import AsyncRetrying

if TYPE_CHECKING:
    from throttled.asyncio import Throttled

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
    """Model which wraps another model such that requests are rate limited with throttled.

    If retryer is provided it also retries requests with tenacity.

    Usage:

    ```python
    from tenacity import AsyncRetrying, stop_after_attempt
    from throttled.asyncio import RateLimiterType, Throttled, rate_limiter, store

    from pydantic_ai import Agent
    from pydantic_ai.models.rate_limited import RateLimitedModel

    throttle = Throttled(
        using=RateLimiterType.GCRA.value,
        quota=rate_limiter.per_sec(1_000, burst=1_000),
        store=store.MemoryStore(),
    )

    model = RateLimitedModel(
        'anthropic:claude-3-7-sonnet-latest',
        limiter=throttle,
        retryer=AsyncRetrying(stop=stop_after_attempt(3)),
    )

    agent = Agent(model=model)
    ```
    """

    def __init__(
        self,
        wrapped: Model | KnownModelName,
        limiter: Throttled | None = None,
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
        key: str = 'default',
        cost: int = 1,
        timeout: float | None = 30.0,
    ) -> ModelResponse:
        """Make a request to the model.

        Args:
            messages: The messages to send to the model.
            model_settings: The settings to use for the model.
            model_request_parameters: The parameters to use for the model.
            key: The key to use in the rate limiter store.
            cost: The cost to use for the rate limiter.
            timeout: The timeout to use for the rate limiter. Important: if timeout is
                not provided or set to -1, the rate limiter will return immediately.
        """
        if self.retryer:
            async for attempt in self.retryer:
                with attempt:
                    if self.limiter:
                        await self.limiter.limit(key, cost, timeout)
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
                await self.limiter.limit(key, cost, timeout)
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
        key: str = 'default',
        cost: int = 1,
        timeout: float | None = 30.0,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streamed request to the model.

        Args:
            messages: The messages to send to the model.
            model_settings: The settings to use for the model.
            model_request_parameters: The parameters to use for the model.
            key: The key to use in the rate limiter store.
            cost: The cost to use for the rate limiter.
            timeout: The timeout to use for the rate limiter. Important: if timeout is
                not provided or set to -1, the rate limiter will return immediately.
        """
        if self.retryer:
            async for attempt in self.retryer:
                with attempt:
                    if self.limiter:
                        await self.limiter.limit(key, cost, timeout)
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
                await self.limiter.limit(key, cost, timeout)
                async with super().request_stream(
                    messages, model_settings, model_request_parameters
                ) as response_stream:
                    yield response_stream
            else:
                async with super().request_stream(
                    messages, model_settings, model_request_parameters
                ) as response_stream:
                    yield response_stream
