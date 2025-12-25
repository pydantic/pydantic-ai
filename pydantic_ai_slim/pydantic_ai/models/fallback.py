from __future__ import annotations as _annotations

import inspect
import warnings
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, NoReturn, get_origin, get_type_hints

from opentelemetry.trace import get_current_span

from pydantic_ai._run_context import RunContext
from pydantic_ai.models.instrumented import InstrumentedModel

from ..exceptions import FallbackExceptionGroup, ModelAPIError
from ..messages import ModelResponse
from ..profiles import ModelProfile
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse, infer_model

if TYPE_CHECKING:
    from ..messages import ModelMessage
    from ..settings import ModelSettings

# Type aliases for handlers
ExceptionHandler = Callable[[Exception], bool]
ResponseHandler = Callable[[ModelResponse], bool]


@dataclass
class OnResponse:
    """Wrapper to explicitly mark a callable as a response handler for fallback_on.

    Most response handlers are auto-detected via type hints. Use this wrapper when:
    - Your handler is a lambda (can't have type hints)
    - Your handler is untyped and you want it treated as a response handler

    Example:
        ```python {test="skip" lint="skip"}
        from pydantic_ai.models.fallback import FallbackModel, OnResponse

        # Auto-detected via type hint (no wrapper needed)
        def check_response(response: ModelResponse) -> bool:
            return 'error' in str(response)

        # Lambda needs explicit wrapper
        fallback_model = FallbackModel(
            model1, model2,
            fallback_on=[ModelAPIError, OnResponse(lambda r: 'error' in str(r))],
        )
        ```
    """

    handler: ResponseHandler

    def __call__(self, response: ModelResponse) -> bool:
        """Call the wrapped handler."""
        return self.handler(response)


# The unified fallback_on type
FallbackOn = (
    type[Exception]
    | tuple[type[Exception], ...]
    | ExceptionHandler
    | ResponseHandler
    | OnResponse
    | Sequence[type[Exception] | ExceptionHandler | ResponseHandler | OnResponse]
)


def _is_response_handler(handler: Callable[..., Any]) -> bool:
    """Check if a callable is a response handler based on type hints.

    Returns True if the first parameter is type-hinted as ModelResponse.
    Returns False otherwise (including if there are no type hints).

    Note: Async response handlers are not supported. If an async callable is detected
    with a ModelResponse type hint, it will still be classified as a response handler,
    but will fail at runtime when called synchronously.
    """
    try:
        # Get the signature to find the first parameter name
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
        if not params:
            return False

        first_param = params[0]

        # Try to get type hints - this resolves forward references automatically
        hints = get_type_hints(handler)
        param_type = hints.get(first_param.name)

        if param_type is None:
            return False

        # Handle parameterized generic types (e.g., Optional[ModelResponse])
        origin = get_origin(param_type)
        check_type = origin if origin is not None else param_type

        # Use identity check or issubclass for robust type comparison
        # This handles forward refs, type aliases, and subclasses correctly
        if check_type is ModelResponse:
            return True
        if isinstance(check_type, type) and issubclass(check_type, ModelResponse):
            return True
        return False
    except Exception:
        # If we can't inspect (e.g., built-in, complex type hints), assume exception handler
        return False


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
                - A response handler (auto-detected by type hint): `def check(r: ModelResponse) -> bool`
                - A response handler (explicit): `OnResponse(lambda r: 'error' in str(r))`
                - A sequence mixing all of the above: `[ModelAPIError, handler, OnResponse(check)]`

                Handler type is auto-detected by inspecting type hints on the first parameter.
                If the first parameter is hinted as `ModelResponse`, it's a response handler.
                Otherwise (including untyped handlers), it's an exception handler.

                Use `OnResponse` wrapper for lambdas or untyped functions that should be
                response handlers.

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

        # Case 2: Single OnResponse wrapper (explicit response handler)
        if isinstance(fallback_on, OnResponse):
            self._response_handlers.append(fallback_on.handler)
            return

        # Case 3: Single exception type
        if _is_exception_type(fallback_on):
            self._exception_handlers.append(_exception_types_to_handler((fallback_on,)))  # type: ignore[arg-type]
            return

        # Case 4: Single callable - auto-detect by type hints
        if callable(fallback_on):
            self._add_handler(fallback_on)
            return

        # Case 5: Sequence of mixed handlers/types
        if isinstance(fallback_on, Sequence) and not isinstance(fallback_on, (str, bytes)):
            for item in fallback_on:
                if isinstance(item, OnResponse):
                    self._response_handlers.append(item.handler)
                elif _is_exception_type(item):
                    self._exception_handlers.append(_exception_types_to_handler((item,)))  # type: ignore[arg-type]
                elif callable(item):
                    self._add_handler(item)
                else:
                    raise TypeError(
                        f'fallback_on items must be exception types, callables, or OnResponse, '
                        f'got {type(item).__name__}'
                    )
            # Warn if empty sequence was provided
            if not self._exception_handlers and not self._response_handlers:
                warnings.warn(
                    'FallbackModel created with empty fallback_on list. '
                    'All exceptions will propagate and all responses will be accepted. '
                    'Consider using fallback_on=(ModelAPIError,) for default behavior.',
                    UserWarning,
                    stacklevel=4,
                )
            return

        raise TypeError(f'Invalid fallback_on type: {type(fallback_on).__name__}')

    def _add_handler(self, handler: Callable[..., Any]) -> None:
        """Add a handler, auto-detecting its type by inspecting type hints."""
        if _is_response_handler(handler):
            self._response_handlers.append(handler)
        else:
            self._exception_handlers.append(handler)

    def _should_fallback_on_exception(self, exc: Exception) -> bool:
        """Check if any exception handler wants to trigger fallback."""
        return any(handler(exc) for handler in self._exception_handlers)

    def _should_fallback_on_response(self, response: ModelResponse) -> bool:
        """Check if any response handler wants to trigger fallback."""
        return any(handler(response) for handler in self._response_handlers)

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
        rejected_models: list[str] = []

        for model in self.models:
            try:
                _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                response = await model.request(messages, model_settings, model_request_parameters)
            except Exception as exc:
                if self._should_fallback_on_exception(exc):
                    exceptions.append(exc)
                    continue
                raise exc

            if self._should_fallback_on_response(response):
                rejected_models.append(model.model_name)
                continue

            self._set_span_attributes(model, prepared_parameters)
            return response

        _raise_fallback_exception_group(exceptions, rejected_models)

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


def _raise_fallback_exception_group(exceptions: list[Exception], rejected_models: list[str]) -> NoReturn:
    """Raise a FallbackExceptionGroup combining exceptions and response rejections.

    Args:
        exceptions: List of exceptions raised by models.
        rejected_models: List of model names whose responses were rejected by fallback_on handlers.
    """
    all_errors: list[Exception] = list(exceptions)
    if rejected_models:
        models_str = ', '.join(f"'{m}'" for m in rejected_models)
        all_errors.append(
            RuntimeError(f'{len(rejected_models)} model response(s) rejected by fallback_on handler: {models_str}')
        )

    if all_errors:
        raise FallbackExceptionGroup('All models from FallbackModel failed', all_errors)
    else:
        raise FallbackExceptionGroup(
            'All models from FallbackModel failed',
            [RuntimeError('No models available')],
        )  # pragma: no cover
