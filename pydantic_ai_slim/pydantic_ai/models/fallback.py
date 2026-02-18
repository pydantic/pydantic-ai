from __future__ import annotations as _annotations

import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard

from opentelemetry.trace import get_current_span
from typing_extensions import assert_never

from pydantic_ai._run_context import RunContext
from pydantic_ai._utils import get_first_param_type, is_async_callable
from pydantic_ai.models.instrumented import InstrumentedModel

from ..exceptions import FallbackExceptionGroup, ModelAPIError
from ..messages import ModelResponse
from ..profiles import ModelProfile
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse, infer_model

if TYPE_CHECKING:
    from ..messages import ModelMessage
    from ..settings import ModelSettings

# Type aliases for handlers (support both sync and async)
_ExceptionHandler = Callable[[Exception], Awaitable[bool]] | Callable[[Exception], bool]
_ResponseHandler = Callable[[ModelResponse], Awaitable[bool]] | Callable[[ModelResponse], bool]


# The unified fallback_on type
_FallbackOn = (
    type[Exception]
    | tuple[type[Exception], ...]
    | _ExceptionHandler
    | _ResponseHandler
    | Sequence[type[Exception] | _ExceptionHandler | _ResponseHandler]
)


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


@dataclass(init=False)
class FallbackModel(Model):
    """A model that uses one or more fallback models upon failure.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    models: list[Model]

    _model_name: str = field(repr=False)
    _exception_handlers: list[_ExceptionHandler] = field(repr=False)
    _response_handlers: list[_ResponseHandler] = field(repr=False)

    def __init__(
        self,
        default_model: Model | KnownModelName | str,
        *fallback_models: Model | KnownModelName | str,
        fallback_on: _FallbackOn = (ModelAPIError,),
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

                Note: For streaming requests, only exception-based fallback is supported, and only for
                errors during stream initialization. Response handlers are ignored for streaming.
                See https://github.com/pydantic/pydantic-ai/issues/4140.
        """
        super().__init__()
        self.models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]

        # Parse fallback_on into exception handlers and response handlers
        self._exception_handlers = []
        self._response_handlers = []
        self._parse_fallback_on(fallback_on)

    def _parse_fallback_on(self, fallback_on: _FallbackOn) -> None:
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

        # Warn if no handlers were registered (empty tuple, empty list, etc.)
        if not self._exception_handlers and not self._response_handlers:
            warnings.warn(
                'FallbackModel created with empty fallback_on. '
                'All exceptions will propagate and all responses will be accepted. '
                'Consider using fallback_on=(ModelAPIError,) for default behavior.',
                UserWarning,
                stacklevel=3,  # user(3) -> __init__(2) -> _parse_fallback_on(1)
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

        Note: For streaming, only exception-based fallback is currently supported,
        and only for errors during stream initialization. Response handlers are
        ignored for streaming requests.
        See https://github.com/pydantic/pydantic-ai/issues/4140.
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


def _exception_types_to_handler(exceptions: tuple[type[Exception], ...]) -> _ExceptionHandler:
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
        all_errors.append(RuntimeError(f'{len(rejected_responses)} model response(s) rejected by fallback_on handler'))
    raise FallbackExceptionGroup('All models from FallbackModel failed', all_errors)
