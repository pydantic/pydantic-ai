from __future__ import annotations as _annotations

import inspect
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard, cast, get_origin

from opentelemetry.trace import get_current_span
from pydantic._internal import _decorators, _typing_extra
from typing_extensions import assert_never

from pydantic_ai._run_context import RunContext
from pydantic_ai.models.instrumented import InstrumentedModel

from ..exceptions import FallbackExceptionGroup, ModelAPIError
from ..messages import ModelResponse
from ..profiles import ModelProfile
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse, infer_model

if TYPE_CHECKING:
    from ..messages import ModelMessage
    from ..settings import ModelSettings

# Type aliases for handlers (support both sync and async)
ExceptionHandler = Callable[[Exception], bool | Awaitable[bool]]
ResponseHandler = Callable[[ModelResponse], bool | Awaitable[bool]]


# The unified fallback_on type
FallbackOn = (
    type[Exception]
    | tuple[type[Exception], ...]
    | ExceptionHandler
    | ResponseHandler
    | Sequence[type[Exception] | ExceptionHandler | ResponseHandler]
)


def _is_response_handler(handler: Callable[..., Any]) -> bool:
    """Check if a callable is a response handler based on type hints.

    Returns True if the first parameter is type-hinted as ModelResponse.
    Returns False otherwise (including if there are no type hints).

    Uses patterns from pydantic_ai._function_schema._takes_ctx for robust type hint detection.
    """
    try:
        sig = inspect.signature(handler)
    except ValueError:
        return False

    try:
        first_param_name = next(iter(sig.parameters.keys()))
    except StopIteration:
        return False

    # Handle callable classes with __call__ method (same pattern as _takes_ctx)
    callable_for_hints = handler
    if not isinstance(handler, _decorators._function_like):  # pyright: ignore[reportPrivateUsage]
        call_func = getattr(type(handler), '__call__', None)
        if call_func is not None:
            callable_for_hints = call_func
        else:
            return False  # pragma: no cover

    try:
        # Use pydantic's get_function_type_hints which handles forward refs better
        type_hints = _typing_extra.get_function_type_hints(_decorators.unwrap_wrapped_function(callable_for_hints))
    except Exception:
        # If we can't get type hints (e.g., built-in), assume exception handler
        return False

    param_type = type_hints.get(first_param_name)
    if param_type is None:
        return False

    # Only support exact ModelResponse type (no Optional, no subclasses)
    return param_type is ModelResponse or get_origin(param_type) is ModelResponse


def _is_exception_type(value: Any) -> TypeGuard[type[Exception]]:
    """Check if value is a single exception type."""
    return isinstance(value, type) and issubclass(value, BaseException)


def _is_exception_types_tuple(value: Any) -> TypeGuard[tuple[type[Exception], ...]]:
    """Check if value is a tuple of exception types."""
    if not isinstance(value, tuple):
        return False
    return all(_is_exception_type(item) for item in cast(tuple[object, ...], value))


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
                - An exception handler: `lambda exc: isinstance(exc, MyError)`
                - A response handler: `def check(r: ModelResponse) -> bool`
                - A sequence mixing all of the above: `[ModelAPIError, exc_handler, response_handler]`

                Handler type is auto-detected by inspecting type hints on the first parameter.
                If the first parameter is hinted as `ModelResponse`, it's a response handler.
                Otherwise (including untyped handlers and lambdas), it's an exception handler.

                Note: For streaming requests, only exception-based fallback is supported, and only for
                errors during stream initialization. Response handlers are ignored for streaming.
                A future release will add streaming support for response-based fallback.
        """
        super().__init__()
        self.models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]

        # Parse fallback_on into exception handlers and response handlers
        self._exception_handlers = []
        self._response_handlers = []
        self._parse_fallback_on(fallback_on)

    def _parse_fallback_on(self, fallback_on: FallbackOn) -> None:
        """Parse the fallback_on parameter into exception and response handlers."""
        if _is_exception_types_tuple(fallback_on):
            # Tuple of exception types
            self._exception_handlers.append(_exception_types_to_handler(fallback_on))
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
            # Warn if empty sequence was provided
            if not self._exception_handlers and not self._response_handlers:
                warnings.warn(
                    'FallbackModel created with empty fallback_on list. '
                    'All exceptions will propagate and all responses will be accepted. '
                    'Consider using fallback_on=(ModelAPIError,) for default behavior.',
                    UserWarning,
                    stacklevel=4,
                )
        else:
            assert_never(fallback_on)  # type: ignore[arg-type]  # pyright can't narrow str/bytes exclusion

    def _add_handler(self, handler: Callable[..., Any]) -> None:
        """Add a handler, auto-detecting its type by inspecting type hints."""
        if _is_response_handler(handler):
            self._response_handlers.append(handler)
        else:
            self._exception_handlers.append(handler)

    async def _should_fallback_on_exception(self, exc: Exception) -> bool:
        """Check if any exception handler wants to trigger fallback."""
        for handler in self._exception_handlers:
            result = handler(exc)
            if inspect.isawaitable(result):
                result = await result
            if result:
                return True
        return False

    async def _should_fallback_on_response(self, response: ModelResponse) -> bool:
        """Check if any response handler wants to trigger fallback."""
        for handler in self._response_handlers:
            result = handler(response)
            if inspect.isawaitable(result):
                result = await result
            if result:
                return True
        return False

    @property
    def model_name(self) -> str:
        """The model name."""
        return f'fallback:{",".join(model.model_name for model in self.models)}'

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
                if await self._should_fallback_on_exception(exc):
                    exceptions.append(exc)
                    continue
                raise exc

            if await self._should_fallback_on_response(response):
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

        Note: For streaming, only exception-based fallback is currently supported,
        and only for errors during stream initialization. Response handlers are
        ignored for streaming requests. A future release will add streaming support
        for response-based fallback.
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
                    if await self._should_fallback_on_exception(exc):
                        exceptions.append(exc)
                        continue
                    raise exc  # pragma: no cover

                # For streaming, we yield the response directly (no response-based fallback)
                self._set_span_attributes(model, prepared_parameters)
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
    if rejected_responses:
        rejection_error = RuntimeError(f'{len(rejected_responses)} model response(s) rejected by fallback_on handler')
        all_errors: list[Exception] = [*exceptions, rejection_error]
    else:
        all_errors = exceptions

    if all_errors:
        raise FallbackExceptionGroup('All models from FallbackModel failed', all_errors)
    else:
        raise FallbackExceptionGroup(
            'All models from FallbackModel failed',
            [RuntimeError('No models available')],
        )  # pragma: no cover
