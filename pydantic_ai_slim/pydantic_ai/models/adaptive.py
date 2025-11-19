from __future__ import annotations as _annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar

from opentelemetry.trace import get_current_span

from pydantic_ai.models.instrumented import InstrumentedModel

from ..exceptions import FallbackExceptionGroup
from ..profiles import ModelProfile
from . import Model, ModelRequestParameters, StreamedResponse

if TYPE_CHECKING:
    from pydantic_ai._run_context import RunContext

    from ..messages import ModelMessage, ModelResponse
    from ..settings import ModelSettings

StateT = TypeVar('StateT')


@dataclass
class Attempt:
    """Record of a single attempt to use a model."""

    model: Model
    """The model that was attempted."""

    exception: Exception | None
    """The exception raised by the model, if any."""

    timestamp: datetime
    """When the attempt was made."""

    duration: timedelta
    """Duration of the attempt."""


@dataclass
class AdaptiveContext(Generic[StateT]):
    """Context provided to the selector function."""

    state: StateT
    """User-defined state object for the selector."""

    attempts: list[Attempt]
    """History of attempts in this request."""

    messages: list[ModelMessage]
    """The original request messages."""

    model_settings: ModelSettings | None
    """Model settings for this request."""

    model_request_parameters: ModelRequestParameters
    """Model request parameters."""


@dataclass(init=False)
class AdaptiveModel(Model, Generic[StateT]):
    """A model that uses custom logic to select which model to try next.

    Unlike FallbackModel which tries models sequentially, AdaptiveModel gives
    full control over model selection based on rich context including attempts,
    exceptions, and custom state.

    The selector function is called before each attempt and can:
    - Return a Model to try next (can be the same model for retry)
    - Raise an exception to stop trying
    - Use async/await for delays (exponential backoff, etc.)
    - Access custom state via ctx.state
    - Inspect previous attempts via ctx.attempts

    Lifecycle hooks provide clean separation of concerns:
    - on_attempt_failed: Called after each failed attempt. Returns bool to continue (True) or stop (False).
    - on_attempt_succeeded: Called after successful attempt (for metrics, quota tracking)
    """

    _selector: Callable[[AdaptiveContext[StateT]], Model] | Callable[[AdaptiveContext[StateT]], Awaitable[Model]]
    _state: StateT
    _on_attempt_failed: (
        Callable[[StateT, Model, Exception, datetime, timedelta], bool]
        | Callable[[StateT, Model, Exception, datetime, timedelta], Awaitable[bool]]
        | None
    )
    _on_attempt_succeeded: (
        Callable[[StateT, Model, ModelResponse, datetime, timedelta], None]
        | Callable[[StateT, Model, ModelResponse, datetime, timedelta], Awaitable[None]]
        | None
    )

    def __init__(
        self,
        selector: Callable[[AdaptiveContext[StateT]], Model] | Callable[[AdaptiveContext[StateT]], Awaitable[Model]],
        *,
        state: StateT | None = None,
        on_attempt_failed: (
            Callable[[StateT, Model, Exception, datetime, timedelta], bool]
            | Callable[[StateT, Model, Exception, datetime, timedelta], Awaitable[bool]]
            | None
        ) = None,
        on_attempt_succeeded: (
            Callable[[StateT, Model, ModelResponse, datetime, timedelta], None]
            | Callable[[StateT, Model, ModelResponse, datetime, timedelta], Awaitable[None]]
            | None
        ) = None,
    ):
        """Initialize an adaptive model instance.

        Args:
            selector: Sync or async function that selects the next model to try.
                Called before each attempt with context including previous attempts.
                Must return a Model. Raise an exception to stop trying.
                The selector manages its own pool of models (via closure, state, etc.).
            state: State object passed to selector. If None, an empty dict is used.
                Reuse the same AdaptiveModel instance to share state across runs.
                Create new instances for isolated state.
            on_attempt_failed: Optional sync or async hook called after each failed attempt.
                Receives (state, model, exception, timestamp, duration).
                Must return bool: True to continue trying, False to stop and re-raise the exception.
                Use for conditional retry logic, throttling detection, error tracking.
            on_attempt_succeeded: Optional sync or async hook called after successful attempt.
                Receives (state, model, response, timestamp, duration).
                Use for quality tracking, quota deduction, metrics collection.
        """
        super().__init__()
        self._selector = selector
        self._state = state if state is not None else {}  # type: ignore
        self._on_attempt_failed = on_attempt_failed
        self._on_attempt_succeeded = on_attempt_succeeded

    @property
    def model_name(self) -> str:
        """The model name."""
        return 'adaptive'

    @property
    def system(self) -> str:
        return 'adaptive'

    @property
    def base_url(self) -> str | None:
        return None

    @cached_property
    def profile(self) -> ModelProfile:
        raise NotImplementedError('AdaptiveModel does not have its own model profile.')

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        return model_request_parameters

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        return model_settings, model_request_parameters

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Try models based on selector logic until one succeeds."""
        attempts: list[Attempt] = []

        while True:
            # Call selector to get next model
            try:
                model = await self._call_selector(
                    AdaptiveContext(
                        state=self._state,
                        attempts=attempts,
                        messages=messages,
                        model_settings=model_settings,
                        model_request_parameters=model_request_parameters,
                    )
                )
            except Exception as exc:
                # Selector raised an exception - stop trying
                exceptions = [a.exception for a in attempts if a.exception is not None]
                raise FallbackExceptionGroup(
                    'AdaptiveModel selector raised an exception',
                    exceptions + [exc],
                ) from exc

            # Try the selected model
            start_time = datetime.now()
            _, prepared_params = model.prepare_request(model_settings, model_request_parameters)

            try:
                response = await model.request(messages, model_settings, model_request_parameters)
                # Success! Set span attributes and call success hook
                duration = datetime.now() - start_time
                self._set_span_attributes(model, prepared_params)

                if self._on_attempt_succeeded is not None:
                    await self._call_hook(
                        self._on_attempt_succeeded, self._state, model, response, start_time, duration
                    )

                return response
            except Exception as exc:
                # Record the attempt
                duration = datetime.now() - start_time
                attempts.append(
                    Attempt(
                        model=model,
                        exception=exc,
                        timestamp=start_time,
                        duration=duration,
                    )
                )

                # Call failure hook to decide whether to continue
                if self._on_attempt_failed is not None:
                    should_continue = await self._call_hook(
                        self._on_attempt_failed, self._state, model, exc, start_time, duration
                    )
                    if not should_continue:
                        # Hook says stop - re-raise the exception
                        raise exc

                # Continue loop to try again

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Try models based on selector logic until one succeeds."""
        attempts: list[Attempt] = []

        while True:
            # Call selector to get next model
            try:
                model = await self._call_selector(
                    AdaptiveContext(
                        state=self._state,
                        attempts=attempts,
                        messages=messages,
                        model_settings=model_settings,
                        model_request_parameters=model_request_parameters,
                    )
                )
            except Exception as exc:
                # Selector raised an exception - stop trying
                exceptions = [a.exception for a in attempts if a.exception is not None]
                raise FallbackExceptionGroup(
                    'AdaptiveModel selector raised an exception',
                    exceptions + [exc],
                ) from exc

            # Try the selected model
            start_time = datetime.now()
            _, prepared_params = model.prepare_request(model_settings, model_request_parameters)

            async with AsyncExitStack() as stack:
                try:
                    response = await stack.enter_async_context(
                        model.request_stream(messages, model_settings, model_request_parameters, run_context)
                    )
                except Exception as exc:
                    # Record the attempt
                    duration = datetime.now() - start_time
                    attempts.append(
                        Attempt(
                            model=model,
                            exception=exc,
                            timestamp=start_time,
                            duration=duration,
                        )
                    )

                    # Call failure hook to decide whether to continue
                    if self._on_attempt_failed is not None:
                        should_continue = await self._call_hook(
                            self._on_attempt_failed, self._state, model, exc, start_time, duration
                        )
                        if not should_continue:
                            # Hook says stop - re-raise the exception
                            raise exc

                    continue

                # Success! Set span attributes and yield
                duration = datetime.now() - start_time
                self._set_span_attributes(model, prepared_params)

                # Note: We call the success hook here, but the response hasn't been fully consumed yet.
                # For streaming, we don't have access to the final ModelResponse until the stream completes.
                # This is a limitation of the streaming API - the hook is called when streaming starts.
                if self._on_attempt_succeeded is not None:
                    # For streaming, we pass the StreamedResponse wrapper
                    # Users should be aware this is called before the stream is consumed
                    await self._call_hook(
                        self._on_attempt_succeeded, self._state, model, response, start_time, duration
                    )  # type: ignore

                yield response
                return

    async def _call_selector(self, context: AdaptiveContext[StateT]) -> Model:
        """Call the selector function, handling both sync and async."""
        if inspect.iscoroutinefunction(self._selector):
            return await self._selector(context)
        else:
            return self._selector(context)  # type: ignore

    async def _call_hook(self, hook: Callable, *args) -> None:  # type: ignore
        """Call a hook function, handling both sync and async."""
        if inspect.iscoroutinefunction(hook):
            await hook(*args)
        else:
            hook(*args)

    def _set_span_attributes(self, model: Model, model_request_parameters: ModelRequestParameters):
        """Set OpenTelemetry span attributes for the successful model."""
        with suppress(Exception):
            span = get_current_span()
            if span.is_recording():
                attributes = getattr(span, 'attributes', {})
                if attributes.get('gen_ai.request.model') == self.model_name:  # pragma: no branch
                    span.set_attributes(
                        {
                            **InstrumentedModel.model_attributes(model),
                            **InstrumentedModel.model_request_parameters_attributes(model_request_parameters),
                        }
                    )
