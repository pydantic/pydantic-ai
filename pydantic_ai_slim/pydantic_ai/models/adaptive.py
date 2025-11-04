from __future__ import annotations as _annotations

import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from opentelemetry.trace import get_current_span

from pydantic_ai._run_context import RunContext
from pydantic_ai.models.instrumented import InstrumentedModel

from ..exceptions import FallbackExceptionGroup
from ..settings import merge_model_settings
from . import Model, ModelRequestParameters, StreamedResponse

if TYPE_CHECKING:
    from ..messages import ModelMessage, ModelResponse
    from ..settings import ModelSettings

AgentDepsT = TypeVar('AgentDepsT')


@dataclass
class AttemptResult:
    """Record of a single attempt to use a model."""

    model: Model
    """The model that was attempted."""

    exception: Exception | None
    """The exception raised by the model, if any."""

    timestamp: float
    """Unix timestamp when the attempt was made."""

    duration: float
    """Duration of the attempt in seconds."""


@dataclass
class AdaptiveContext(Generic[AgentDepsT]):
    """Context provided to the selector function."""

    run_context: RunContext[AgentDepsT] | None
    """Access to agent dependencies. May be None for non-streaming requests."""

    models: Sequence[Model]
    """Available models to choose from."""

    attempts: list[AttemptResult]
    """History of attempts in this request."""

    attempt_number: int
    """Current attempt number (1-indexed)."""

    messages: list[ModelMessage]
    """The original request messages."""

    model_settings: ModelSettings | None
    """Model settings for this request."""

    model_request_parameters: ModelRequestParameters
    """Model request parameters."""


@dataclass(init=False)
class AdaptiveModel(Model, Generic[AgentDepsT]):
    """A model that uses custom logic to select which model to try next.

    Unlike FallbackModel which tries models sequentially, AdaptiveModel gives
    full control over model selection based on rich context including attempts,
    exceptions, and agent dependencies.

    The selector function is called before each attempt and can:
    - Return a Model to try next (can be the same model for retry)
    - Return None to stop trying
    - Use async/await for delays (exponential backoff, etc.)
    - Access agent dependencies via ctx.run_context.deps
    - Inspect previous attempts via ctx.attempts
    """

    models: Sequence[Model]
    _selector: (
        Callable[[AdaptiveContext[AgentDepsT]], Model | None]
        | Callable[[AdaptiveContext[AgentDepsT]], Awaitable[Model | None]]
    )
    _max_attempts: int | None

    def __init__(
        self,
        models: Sequence[Model],
        selector: Callable[[AdaptiveContext[AgentDepsT]], Model | None]
        | Callable[[AdaptiveContext[AgentDepsT]], Awaitable[Model | None]],
        *,
        max_attempts: int | None = None,
    ):
        """Initialize an adaptive model instance.

        Args:
            models: Pool of models to choose from.
            selector: Sync or async function that selects the next model to try.
                Called before each attempt with context including previous attempts.
                Return a Model to try, or None to stop.
            max_attempts: Maximum total attempts across all models (None = unlimited).
        """
        super().__init__()
        if not models:
            raise ValueError('At least one model must be provided')

        self.models = list(models)
        self._selector = selector
        self._max_attempts = max_attempts

    @property
    def model_name(self) -> str:
        """The model name."""
        return f'adaptive:{",".join(model.model_name for model in self.models)}'

    @property
    def system(self) -> str:
        return f'adaptive:{",".join(model.system for model in self.models)}'

    @property
    def base_url(self) -> str | None:
        return self.models[0].base_url if self.models else None

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Try models based on selector logic until one succeeds or selector returns None."""
        attempts: list[AttemptResult] = []
        attempt_number = 0

        while True:
            attempt_number += 1

            # Check max attempts
            if self._max_attempts is not None and attempt_number > self._max_attempts:
                exceptions = [a.exception for a in attempts if a.exception is not None]
                if exceptions:
                    raise FallbackExceptionGroup(
                        f'AdaptiveModel exceeded max_attempts of {self._max_attempts}', exceptions
                    )
                else:
                    raise FallbackExceptionGroup(
                        f'AdaptiveModel exceeded max_attempts of {self._max_attempts}',
                        [RuntimeError('No models were attempted')],
                    )

            # Create context for selector
            context = AdaptiveContext(
                run_context=None,  # run_context not available in non-streaming request
                models=self.models,
                attempts=attempts,
                attempt_number=attempt_number,
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
            )

            # Call selector to get next model
            model = await self._call_selector(context)

            if model is None:
                # Selector says stop trying
                exceptions = [a.exception for a in attempts if a.exception is not None]
                if exceptions:
                    raise FallbackExceptionGroup('AdaptiveModel selector returned None', exceptions)
                else:
                    raise FallbackExceptionGroup(
                        'AdaptiveModel selector returned None', [RuntimeError('No models were attempted')]
                    )

            # Try the selected model
            start_time = time.time()
            customized_params = model.customize_request_parameters(model_request_parameters)
            merged_settings = merge_model_settings(model.settings, model_settings)

            try:
                response = await model.request(messages, merged_settings, customized_params)
                # Success! Set span attributes and return
                self._set_span_attributes(model)
                return response
            except Exception as exc:
                # Record the attempt
                duration = time.time() - start_time
                attempts.append(
                    AttemptResult(
                        model=model,
                        exception=exc,
                        timestamp=start_time,
                        duration=duration,
                    )
                )
                # Continue loop to try again

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[AgentDepsT] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Try models based on selector logic until one succeeds or selector returns None."""
        attempts: list[AttemptResult] = []
        attempt_number = 0

        while True:
            attempt_number += 1

            # Check max attempts
            if self._max_attempts is not None and attempt_number > self._max_attempts:
                exceptions = [a.exception for a in attempts if a.exception is not None]
                if exceptions:
                    raise FallbackExceptionGroup(
                        f'AdaptiveModel exceeded max_attempts of {self._max_attempts}', exceptions
                    )
                else:
                    raise FallbackExceptionGroup(
                        f'AdaptiveModel exceeded max_attempts of {self._max_attempts}',
                        [RuntimeError('No models were attempted')],
                    )

            # Create context for selector
            context = AdaptiveContext(
                run_context=run_context,
                models=self.models,
                attempts=attempts,
                attempt_number=attempt_number,
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
            )

            # Call selector to get next model
            model = await self._call_selector(context)

            if model is None:
                # Selector says stop trying
                exceptions = [a.exception for a in attempts if a.exception is not None]
                if exceptions:
                    raise FallbackExceptionGroup('AdaptiveModel selector returned None', exceptions)
                else:
                    raise FallbackExceptionGroup(
                        'AdaptiveModel selector returned None', [RuntimeError('No models were attempted')]
                    )

            # Try the selected model
            start_time = time.time()
            customized_params = model.customize_request_parameters(model_request_parameters)
            merged_settings = merge_model_settings(model.settings, model_settings)

            async with AsyncExitStack() as stack:
                try:
                    response = await stack.enter_async_context(
                        model.request_stream(messages, merged_settings, customized_params, run_context)
                    )
                except Exception as exc:
                    # Record the attempt and continue
                    duration = time.time() - start_time
                    attempts.append(
                        AttemptResult(
                            model=model,
                            exception=exc,
                            timestamp=start_time,
                            duration=duration,
                        )
                    )
                    continue

                # Success! Set span attributes and yield
                self._set_span_attributes(model)
                yield response
                return

    async def _call_selector(self, context: AdaptiveContext[AgentDepsT]) -> Model | None:
        """Call the selector function, handling both sync and async."""
        if inspect.iscoroutinefunction(self._selector):
            return await self._selector(context)
        else:
            return self._selector(context)  # type: ignore

    def _set_span_attributes(self, model: Model):
        """Set OpenTelemetry span attributes for the successful model."""
        with suppress(Exception):
            span = get_current_span()
            if span.is_recording():
                attributes = getattr(span, 'attributes', {})
                if attributes.get('gen_ai.request.model') == self.model_name:  # pragma: no branch
                    span.set_attributes(InstrumentedModel.model_attributes(model))
