"""Concurrency limiting wrapper for models."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from .._run_context import RunContext, get_current_run_context
from ..concurrency import (
    AbstractConcurrencyLimiter,
    AnyConcurrencyLimit,
    ConcurrencyLimit,
    ConcurrencyLimiter,
    get_concurrency_context,
    normalize_to_limiter,
)
from ..exceptions import UserError
from ..messages import ModelMessage, ModelResponse
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import KnownModelName, Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from .wrapper import WrapperModel

_ACTIVE_MODEL_LIMITERS: ContextVar[tuple[AbstractConcurrencyLimiter, ...]] = ContextVar(
    'pydantic_ai.active_model_limiters', default=()
)


@contextmanager
def _active_model_limiter(limiter: AbstractConcurrencyLimiter) -> Generator[None]:
    token = _ACTIVE_MODEL_LIMITERS.set((*_ACTIVE_MODEL_LIMITERS.get(), limiter))
    try:
        yield
    finally:
        _ACTIVE_MODEL_LIMITERS.reset(token)


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

    # Or share a limiter across multiple models
    from pydantic_ai import ConcurrencyLimiter  # noqa E402

    shared_limiter = ConcurrencyLimiter(max_running=10, name='openai-pool')
    model1 = ConcurrencyLimitedModel('openai:gpt-4o', limiter=shared_limiter)
    model2 = ConcurrencyLimitedModel('openai:gpt-4o-mini', limiter=shared_limiter)
    ```
    """

    _limiter: AbstractConcurrencyLimiter

    def __init__(
        self,
        wrapped: Model | KnownModelName,
        limiter: int | ConcurrencyLimit | AbstractConcurrencyLimiter,
    ):
        """Initialize the ConcurrencyLimitedModel.

        Args:
            wrapped: The model to wrap, either a Model instance or a known model name.
            limiter: The concurrency limit configuration. Can be:
                - An `int`: Simple limit on concurrent operations (unlimited queue).
                - A `ConcurrencyLimit`: Full configuration with optional backpressure.
                - An `AbstractConcurrencyLimiter`: A pre-created limiter for sharing across models.
        """
        super().__init__(wrapped)
        if isinstance(limiter, AbstractConcurrencyLimiter):
            self._limiter = limiter
        else:
            self._limiter = ConcurrencyLimiter.from_limit(limiter)

    def _ensure_distinct_limiter(self, run_context: RunContext[Any] | None = None) -> None:
        if any(limiter is self._limiter for limiter in _ACTIVE_MODEL_LIMITERS.get()):
            raise UserError(
                'Nested `ConcurrencyLimitedModel` wrappers use the same concurrency limiter, which can deadlock '
                'a request. Use separate limiters or wrap the model only once.'
            )
        run_context = run_context or get_current_run_context()
        agent = run_context.agent if run_context is not None else None
        if agent is not None and agent._concurrency_limiter is self._limiter:  # pyright: ignore[reportPrivateUsage]
            raise UserError(
                'The agent and a `ConcurrencyLimitedModel` use the same concurrency limiter, which can deadlock '
                'a request. Use separate limiters or apply concurrency limiting at only one layer.'
            )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the model with concurrency limiting."""
        self._ensure_distinct_limiter()
        async with get_concurrency_context(self._limiter, f'model:{self.model_name}'):
            with _active_model_limiter(self._limiter):
                return await self.wrapped.request(messages, model_settings, model_request_parameters)

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        """Count tokens with concurrency limiting."""
        self._ensure_distinct_limiter()
        async with get_concurrency_context(self._limiter, f'model:{self.model_name}'):
            with _active_model_limiter(self._limiter):
                return await self.wrapped.count_tokens(messages, model_settings, model_request_parameters)

    async def compact_messages(
        self, request_context: ModelRequestContext, *, instructions: str | None = None
    ) -> ModelResponse:
        """Compact messages with concurrency limiting."""
        self._ensure_distinct_limiter()
        async with get_concurrency_context(self._limiter, f'model:{self.model_name}'):
            with _active_model_limiter(self._limiter):
                return await self.wrapped.compact_messages(request_context, instructions=instructions)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        """Make a streaming request to the model with concurrency limiting."""
        self._ensure_distinct_limiter(run_context)
        async with get_concurrency_context(self._limiter, f'model:{self.model_name}'):
            async with AsyncExitStack() as stack:
                # Open nested wrappers while the limiter is visible, then restore this task's context
                # before yielding: the stream may be closed on another task.
                with _active_model_limiter(self._limiter):
                    response_stream = await stack.enter_async_context(
                        self.wrapped.request_stream(messages, model_settings, model_request_parameters, run_context)
                    )
                yield response_stream


def limit_model_concurrency(
    model: Model | KnownModelName,
    limiter: AnyConcurrencyLimit,
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
    from pydantic_ai.models.concurrency import limit_model_concurrency

    model = limit_model_concurrency('openai:gpt-4o', limiter=5)
    ```
    """
    normalized_limiter = normalize_to_limiter(limiter)
    if normalized_limiter is None:
        from . import infer_model

        return infer_model(model) if isinstance(model, str) else model
    return ConcurrencyLimitedModel(model, normalized_limiter)
