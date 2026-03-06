from __future__ import annotations as _annotations

import dataclasses
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard

from opentelemetry.trace import get_current_span
from typing_extensions import assert_never

from pydantic_ai._run_context import RunContext
from pydantic_ai._utils import get_first_param_type, is_async_callable
from pydantic_ai.models.instrumented import InstrumentedModel

from ..exceptions import FallbackExceptionGroup, ModelAPIError, UserError
from ..messages import ModelResponse, ModelResponseStreamEvent
from ..profiles import ModelProfile
from ..usage import RequestUsage
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse, infer_model

if TYPE_CHECKING:
    from ..messages import ModelMessage
    from ..settings import ModelSettings

ExceptionHandler = Callable[[Exception], Awaitable[bool]] | Callable[[Exception], bool]
"""A sync or async callable that decides whether an exception should trigger fallback."""

ResponseHandler = Callable[[ModelResponse], Awaitable[bool]] | Callable[[ModelResponse], bool]
"""A sync or async callable that decides whether a model response should trigger fallback."""

FallbackOn = (
    type[Exception]
    | tuple[type[Exception], ...]
    | ExceptionHandler
    | ResponseHandler
    | Sequence[type[Exception] | ExceptionHandler | ResponseHandler]
)
"""The type of the `fallback_on` parameter to [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel]."""


class ResponseRejected(Exception):
    """Raised within a `FallbackExceptionGroup` when model responses are rejected by a response handler."""

    def __init__(self, rejected_count: int):
        super().__init__(f'{rejected_count} model response(s) rejected by fallback_on handler')


def _is_response_handler(handler: Callable[..., Any]) -> bool:
    """Check if a callable is a response handler based on type hints.

    Returns True if the first parameter is type-hinted as ModelResponse.
    Returns False otherwise (including if there are no type hints).
    """
    first_param_type = get_first_param_type(handler)
    if first_param_type is None:
        return False
    # Only support exact ModelResponse type (no Optional, no subclasses)
    return first_param_type is ModelResponse


def _is_exception_type(value: Any) -> TypeGuard[type[Exception]]:
    """Check if value is a single exception type."""
    return isinstance(value, type) and issubclass(value, Exception)


class _ResponseCheckingStream(StreamedResponse):
    """Wraps a StreamedResponse to check response handlers after stream consumption.

    Proxies all attributes to the wrapped stream via __getattr__.
    After all events are yielded, calls the response handler and raises
    ResponseRejected if the response is rejected.
    """

    def __init__(
        self,
        wrapped: StreamedResponse,
        should_fallback: Callable[[Exception | ModelResponse], Awaitable[bool]],
    ):
        # Skip super().__init__() so dataclass field defaults don't create instance attributes
        object.__setattr__(self, '_wrapped', wrapped)
        object.__setattr__(self, '_should_fallback', should_fallback)
        object.__setattr__(self, '_cached_iter', None)

    # Dataclass fields with default=None create class-level attributes that shadow __getattr__.
    # Derived from StreamedResponse's fields so new fields are automatically included.
    _PROXIED_FIELDS = frozenset(
        f.name for f in dataclasses.fields(StreamedResponse) if not f.init and f.default is not dataclasses.MISSING
    )

    def __getattribute__(self, name: str) -> Any:
        # For proxied fields, go straight to the wrapped stream instead of finding
        # the class-level None default inherited from StreamedResponse's dataclass.
        if name in object.__getattribute__(self, '_PROXIED_FIELDS'):
            return getattr(object.__getattribute__(self, '_wrapped'), name)
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Redirect writes to the wrapped stream (e.g. final_result_event set by agent loop)
        if name in ('_wrapped', '_should_fallback', '_cached_iter'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped, name, value)

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        if self._cached_iter is None:
            object.__setattr__(self, '_cached_iter', self._iterate())
        return self._cached_iter

    async def _iterate(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for event in self._wrapped:
            yield event
        # Stream fully consumed — check response handlers
        response = self._wrapped.get()
        if await self._should_fallback(response):
            raise ResponseRejected(1)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        # Not called — __aiter__ is overridden to proxy the wrapped stream directly
        raise NotImplementedError  # pragma: no cover
        yield  # pragma: no cover

    def get(self) -> ModelResponse:
        return self._wrapped.get()

    def usage(self) -> RequestUsage:
        return self._wrapped.usage()

    @property
    def model_name(self) -> str:
        return self._wrapped.model_name

    @property
    def provider_name(self) -> str | None:
        return self._wrapped.provider_name

    @property
    def provider_url(self) -> str | None:
        return self._wrapped.provider_url

    @property
    def timestamp(self) -> datetime:
        return self._wrapped.timestamp


@dataclass(init=False)
class FallbackModel(Model):
    """A model that uses one or more fallback models upon failure.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    models: list[Model]

    _model_name: str = field(repr=False)
    _exception_handlers: list[ExceptionHandler] = field(repr=False)
    _response_handlers: list[ResponseHandler] = field(repr=False)

    def __init__(
        self,
        default_model: Model | KnownModelName | str,
        *fallback_models: Model | KnownModelName | str,
        fallback_on: FallbackOn = (ModelAPIError,),
    ):
        """Initialize a fallback model instance.

        Args:
            default_model: The name or instance of the default model to use.
            fallback_models: The names or instances of the fallback models to use upon failure.
            fallback_on: Conditions that trigger fallback to the next model. Accepts:

                - A tuple of exception types: `(ModelAPIError, RateLimitError)`
                - An exception handler (sync or async): `lambda exc: isinstance(exc, MyError)`
                - A response handler (sync or async): `def check(r: ModelResponse) -> bool`
                - A sequence mixing all of the above: `[ModelAPIError, exc_handler, response_handler]`

                Handler type is auto-detected by inspecting type hints on the first parameter.
                If the first parameter is hinted as `ModelResponse`, it's a response handler.
                Otherwise (including untyped handlers and lambdas), it's an exception handler.
        """
        super().__init__()
        self.models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]

        # Parse fallback_on into exception handlers and response handlers
        self._exception_handlers = []
        self._response_handlers = []
        self._parse_fallback_on(fallback_on)

    def _parse_fallback_on(self, fallback_on: FallbackOn) -> None:
        """Parse the fallback_on parameter into exception and response handlers."""
        if isinstance(fallback_on, tuple):
            if fallback_on:
                # Tuple of exception types (typing guarantees tuple contents are exception types)
                self._exception_handlers.append(_exception_types_to_handler(fallback_on))  # type: ignore[arg-type]
        elif _is_exception_type(fallback_on):
            # Single exception type
            self._exception_handlers.append(_exception_types_to_handler((fallback_on,)))
        elif callable(fallback_on):
            # Single callable - auto-detect by type hints
            self._add_handler(fallback_on)
        elif isinstance(fallback_on, Sequence) and not isinstance(fallback_on, (str, bytes)):
            # Sequence of mixed handlers/types
            for item in fallback_on:
                if _is_exception_type(item):
                    self._exception_handlers.append(_exception_types_to_handler((item,)))
                elif callable(item):
                    self._add_handler(item)
                else:
                    # Types guarantee all items are exception types or callables
                    assert_never(item)
        else:
            assert_never(fallback_on)  # type: ignore[arg-type]  # pyright can't narrow str/bytes exclusion

        if not self._exception_handlers and not self._response_handlers:
            raise UserError(
                'FallbackModel created with empty fallback_on. '
                'All exceptions will propagate and all responses will be accepted. '
                'Use fallback_on=(ModelAPIError,) for default behavior.'
            )

    def _add_handler(self, handler: Callable[..., Any]) -> None:
        """Add a handler, auto-detecting its type by inspecting type hints."""
        if _is_response_handler(handler):
            self._response_handlers.append(handler)
        else:
            self._exception_handlers.append(handler)

    async def _should_fallback(self, value: Exception | ModelResponse) -> bool:
        """Check if any handler wants to trigger fallback."""
        handlers = self._exception_handlers if isinstance(value, Exception) else self._response_handlers
        for handler in handlers:
            # pyright can't narrow handler's param type from the isinstance check on value
            result = await handler(value) if is_async_callable(handler) else handler(value)  # type: ignore[arg-type]
            if result:
                return True
        return False

    @property
    def model_name(self) -> str:
        """The model name."""
        return f'fallback:{",".join(model.model_name for model in self.models)}'

    @property
    def model_id(self) -> str:
        """The fully qualified model identifier, combining the wrapped models' IDs."""
        return f'fallback:{",".join(model.model_id for model in self.models)}'

    @property
    def system(self) -> str:
        return f'fallback:{",".join(model.system for model in self.models)}'

    @property
    def base_url(self) -> str | None:
        return self.models[0].base_url

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Try each model in sequence until one succeeds.

        In case of failure, raise a FallbackExceptionGroup with all exceptions.
        """
        exceptions: list[Exception] = []
        rejected_responses: list[ModelResponse] = []

        for model in self.models:
            try:
                _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                response = await model.request(messages, model_settings, model_request_parameters)
            except Exception as exc:
                if await self._should_fallback(exc):
                    exceptions.append(exc)
                    continue
                raise exc

            if await self._should_fallback(response):
                rejected_responses.append(response)
                continue

            self._set_span_attributes(model, prepared_parameters)
            return response

        _raise_fallback_exception_group(exceptions, rejected_responses)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Try each model in sequence until one succeeds.

        Response handlers are checked after stream consumption. If the response
        is rejected, a `ResponseRejected` is raised (wrapped in a `FallbackExceptionGroup`).
        Unlike non-streaming requests, transparent retry with the next model is not possible
        because events have already been yielded to the caller.
        """
        exceptions: list[Exception] = []

        for model in self.models:
            async with AsyncExitStack() as stack:
                try:
                    _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                    response = await stack.enter_async_context(
                        model.request_stream(messages, model_settings, model_request_parameters, run_context)
                    )
                except Exception as exc:
                    if await self._should_fallback(exc):
                        exceptions.append(exc)
                        continue
                    raise exc  # pragma: no cover

                self._set_span_attributes(model, prepared_parameters)
                if self._response_handlers:
                    try:
                        yield _ResponseCheckingStream(response, self._should_fallback)
                    except ResponseRejected as e:
                        # Response was rejected after stream consumption.
                        # Can't retry — events already consumed by caller.
                        exceptions.append(e)
                        break
                else:
                    yield response
                return

        _raise_fallback_exception_group(exceptions, [])

    @cached_property
    def profile(self) -> ModelProfile:
        raise NotImplementedError('FallbackModel does not have its own model profile.')

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        return model_request_parameters  # pragma: no cover

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        return model_settings, model_request_parameters

    def _set_span_attributes(self, model: Model, model_request_parameters: ModelRequestParameters) -> None:
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


def _exception_types_to_handler(exceptions: tuple[type[Exception], ...]) -> ExceptionHandler:
    """Create an exception handler from a tuple of exception types."""

    def handler(exc: Exception) -> bool:
        return isinstance(exc, exceptions)

    return handler


def _raise_fallback_exception_group(exceptions: list[Exception], rejected_responses: list[ModelResponse]) -> NoReturn:
    """Raise a FallbackExceptionGroup combining exceptions and response rejections.

    Args:
        exceptions: List of exceptions raised by models.
        rejected_responses: List of responses that were rejected by fallback_on handlers.
    """
    all_errors = list(exceptions)
    if rejected_responses:
        all_errors.append(ResponseRejected(len(rejected_responses)))
    raise FallbackExceptionGroup('All models from FallbackModel failed', all_errors)
