from __future__ import annotations as _annotations

import inspect
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, NoReturn

from opentelemetry.trace import get_current_span

from pydantic_ai._run_context import RunContext
from pydantic_ai.models.instrumented import InstrumentedModel

from ..exceptions import FallbackExceptionGroup, ModelAPIError
from ..profiles import ModelProfile
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse, infer_model

if TYPE_CHECKING:
    from ..messages import ModelMessage, ModelResponse
    from ..settings import ModelSettings

# Type aliases for handlers
ExceptionHandler = Callable[[Exception], bool]
ResponseHandler = Callable[['ModelResponse', list['ModelMessage']], bool]

# The unified fallback_on type
FallbackOn = (
    tuple[type[Exception], ...]
    | ExceptionHandler
    | ResponseHandler
    | Sequence[type[Exception] | ExceptionHandler | ResponseHandler]
)


def _get_required_param_count(handler: Callable[..., Any]) -> int:
    """Count required positional parameters of a callable."""
    try:
        sig = inspect.signature(handler)
        return sum(
            1
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
    except (ValueError, TypeError):
        # If we can't inspect (e.g., built-in), assume exception handler
        return 1  # pragma: no cover


def _is_exception_types_tuple(value: Any) -> bool:
    """Check if value is a tuple of exception types (the old API format)."""
    if not isinstance(value, tuple):
        return False
    items: tuple[Any, ...] = value
    for item in items:
        if not isinstance(item, type) or not issubclass(item, BaseException):
            return False
    return True


def _is_exception_type(value: Any) -> bool:
    """Check if value is a single exception type."""
    return isinstance(value, type) and issubclass(value, BaseException)


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
                - A response handler: `lambda response, messages: 'error' in str(response)`
                - A sequence mixing all of the above: `[ModelAPIError, my_handler, response_check]`

                Exception handlers take one parameter (the exception) and return `True` to trigger fallback.
                Response handlers take two parameters (ModelResponse, list[ModelMessage]) and return `True`
                to trigger fallback based on response content.

                Handler type is auto-detected by parameter count: 1 param = exception handler,
                2 params = response handler.

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
        # Case 1: Tuple of exception types (backward compatible)
        if _is_exception_types_tuple(fallback_on):
            self._exception_handlers.append(_exception_types_to_handler(fallback_on))  # type: ignore[arg-type]
            return

        # Case 2: Single callable
        if callable(fallback_on):
            self._add_handler(fallback_on)
            return

        # Case 3: Sequence of mixed handlers/types
        if isinstance(fallback_on, Sequence) and not isinstance(fallback_on, (str, bytes)):
            for item in fallback_on:
                if _is_exception_type(item):
                    # Single exception type - wrap in handler
                    self._exception_handlers.append(_exception_types_to_handler((item,)))  # type: ignore[arg-type]
                elif callable(item):
                    self._add_handler(item)
                else:
                    raise TypeError(
                        f'fallback_on items must be exception types or callables, got {type(item).__name__}'
                    )
            return

        raise TypeError(f'Invalid fallback_on type: {type(fallback_on).__name__}')

    def _add_handler(self, handler: Callable[..., Any]) -> None:
        """Add a handler, auto-detecting its type by parameter count."""
        param_count = _get_required_param_count(handler)
        if param_count >= 2:
            self._response_handlers.append(handler)
        else:
            self._exception_handlers.append(handler)

    def _should_fallback_on_exception(self, exc: Exception) -> bool:
        """Check if any exception handler wants to trigger fallback."""
        return any(handler(exc) for handler in self._exception_handlers)

    def _should_fallback_on_response(self, response: ModelResponse, messages: list[ModelMessage]) -> bool:
        """Check if any response handler wants to trigger fallback."""
        return any(handler(response, messages) for handler in self._response_handlers)

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
        response_rejections: int = 0

        for model in self.models:
            try:
                _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                response = await model.request(messages, model_settings, model_request_parameters)
            except Exception as exc:
                if self._should_fallback_on_exception(exc):
                    exceptions.append(exc)
                    continue
                raise exc

            if self._should_fallback_on_response(response, messages):
                response_rejections += 1
                continue

            self._set_span_attributes(model, prepared_parameters)
            return response

        _raise_fallback_exception_group(exceptions, response_rejections)

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
                    if self._should_fallback_on_exception(exc):
                        exceptions.append(exc)
                        continue
                    raise exc  # pragma: no cover

                # For streaming, we yield the response directly (no response-based fallback)
                self._set_span_attributes(model, prepared_parameters)
                yield response
                return

        _raise_fallback_exception_group(exceptions, 0)

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


def _raise_fallback_exception_group(exceptions: list[Exception], response_rejections: int) -> NoReturn:
    """Raise a FallbackExceptionGroup combining exceptions and rejections."""
    all_errors: list[Exception] = list(exceptions)
    if response_rejections > 0:
        all_errors.append(RuntimeError(f'{response_rejections} model response(s) rejected by fallback_on'))

    if all_errors:
        raise FallbackExceptionGroup('All models from FallbackModel failed', all_errors)
    else:
        raise FallbackExceptionGroup(
            'All models from FallbackModel failed',
            [RuntimeError('No models available')],
        )  # pragma: no cover
