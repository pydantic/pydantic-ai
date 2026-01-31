"""Concurrency limiting wrapper for models."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from .._run_context import RunContext
from ..concurrency import (
    ConcurrencyLimit,
    ConcurrencyLimiter,
    RecordingSemaphore,
    get_concurrency_context,
    normalize_to_semaphore,
)
from ..messages import ModelMessage, ModelResponse
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse
from .wrapper import WrapperModel


@dataclass(init=False)
class ConcurrencyLimitedModel(WrapperModel):
    """A model wrapper that limits concurrent requests to the underlying model.

    This wrapper applies concurrency limiting at the model level, ensuring that
    the number of concurrent requests to the model does not exceed the configured
    limit. This is useful for:

    - Respecting API rate limits
    - Managing resource usage
    - Sharing a concurrency pool across multiple models

    Example usage:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

    # Limit to 5 concurrent requests
    model = ConcurrencyLimitedModel('openai:gpt-4o', limiter=5)
    agent = Agent(model)

    # Or share a semaphore across multiple models
    from pydantic_ai.concurrency import RecordingSemaphore

    shared_semaphore = RecordingSemaphore(max_running=10, name='openai-pool')
    model1 = ConcurrencyLimitedModel('openai:gpt-4o', limiter=shared_semaphore)
    model2 = ConcurrencyLimitedModel('openai:gpt-4o-mini', limiter=shared_semaphore)
    ```
    """

    _semaphore: RecordingSemaphore

    def __init__(
        self,
        wrapped: Model | KnownModelName,
        limiter: int | ConcurrencyLimiter | RecordingSemaphore,
    ):
        """Initialize the ConcurrencyLimitedModel.

        Args:
            wrapped: The model to wrap, either a Model instance or a known model name.
            limiter: The concurrency limit configuration. Can be:
                - An `int`: Simple limit on concurrent operations (unlimited queue).
                - A `ConcurrencyLimiter`: Full configuration with optional backpressure.
                - A `RecordingSemaphore`: A pre-created semaphore for sharing across models.
        """
        super().__init__(wrapped)
        if isinstance(limiter, RecordingSemaphore):
            self._semaphore = limiter
        else:
            self._semaphore = RecordingSemaphore.from_limit(limiter)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the model with concurrency limiting."""
        async with get_concurrency_context(self._semaphore, f'model:{self.model_name}'):
            return await self.wrapped.request(messages, model_settings, model_request_parameters)

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        """Count tokens with concurrency limiting."""
        async with get_concurrency_context(self._semaphore, f'model:{self.model_name}'):
            return await self.wrapped.count_tokens(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to the model with concurrency limiting."""
        async with get_concurrency_context(self._semaphore, f'model:{self.model_name}'):
            async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as response_stream:
                yield response_stream


def limit_model_concurrency(
    model: Model | KnownModelName,
    limiter: ConcurrencyLimit,
) -> Model:
    """Wrap a model with concurrency limiting.

    This is a convenience function to wrap a model with concurrency limiting.
    If the limiter is None, the model is returned unchanged.

    Args:
        model: The model to wrap.
        limiter: The concurrency limit configuration.

    Returns:
        The wrapped model with concurrency limiting, or the original model if limiter is None.

    Example:
    ```python
    from pydantic_ai.models import infer_model
    from pydantic_ai.models.concurrency import limit_model_concurrency

    model = limit_model_concurrency('openai:gpt-4o', limiter=5)
    ```
    """
    semaphore = normalize_to_semaphore(limiter)
    if semaphore is None:
        from . import infer_model

        return infer_model(model) if isinstance(model, str) else model
    return ConcurrencyLimitedModel(model, semaphore)
